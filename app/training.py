import logging
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import json
import queue
import os
import signal
import psutil

DEFAULT_OUTPUT_DIR = "outputs/train"
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

# tqdm progress: "Training:   1%|▏         | 125/10000 [02:02<2:36:10,  1.05step/s]"
# Captures: current_step, total_steps, eta string (HH:MM:SS or MM:SS).
_TQDM_RE = re.compile(
    r"Training:\s*\d+%[^|]*\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:[\d:]+)<([\d:]+)"
)


def _parse_duration(s: str) -> Optional[float]:
    """Parse tqdm's HH:MM:SS or MM:SS into seconds. Returns None on '?'."""
    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def _generate_output_dir(policy_type: str, dataset_repo_id: str) -> str:
    """Build a sortable, collision-free path under outputs/train/.

    LeRobot refuses to write into an existing directory, so each run needs a
    unique leaf. Timestamp + policy + dataset slug makes runs discoverable on
    disk for later inference without needing a metadata DB.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_slug = _SLUG_RE.sub("_", dataset_repo_id).strip("_") or "dataset"
    return f"{DEFAULT_OUTPUT_DIR}/{policy_type}_{dataset_slug}_{timestamp}"

logger = logging.getLogger(__name__)

class TrainingRequest(BaseModel):
    # Dataset configuration - exact matches from CLI
    dataset_repo_id: str  # --dataset.repo_id
    dataset_revision: Optional[str] = None  # --dataset.revision
    dataset_root: Optional[str] = None  # --dataset.root
    dataset_episodes: Optional[List[int]] = None  # --dataset.episodes
    
    # Policy configuration - only type is configurable at top level
    policy_type: str = "act"  # --policy.type (act, diffusion, pi0, smolvla, tdmpc, vqbet, pi0fast, sac, reward_classifier)
    
    # Core training parameters - exact matches from CLI
    steps: int = 10000  # --steps
    batch_size: int = 8  # --batch_size
    seed: Optional[int] = 1000  # --seed
    num_workers: int = 4  # --num_workers
    
    # Logging and checkpointing - exact matches from CLI
    log_freq: int = 250  # --log_freq
    save_freq: int = 1000  # --save_freq
    eval_freq: int = 0  # --eval_freq
    save_checkpoint: bool = True  # --save_checkpoint
    
    # Output configuration - exact matches from CLI
    output_dir: str = "outputs/train"  # --output_dir
    resume: bool = False  # --resume
    job_name: Optional[str] = None  # --job_name
    
    # Weights & Biases - exact matches from CLI
    wandb_enable: bool = False  # --wandb.enable
    wandb_project: Optional[str] = None  # --wandb.project
    wandb_entity: Optional[str] = None  # --wandb.entity
    wandb_notes: Optional[str] = None  # --wandb.notes
    wandb_run_id: Optional[str] = None  # --wandb.run_id
    wandb_mode: Optional[str] = "online"  # --wandb.mode (online, offline, disabled)
    wandb_disable_artifact: bool = False  # --wandb.disable_artifact
    
    # Environment and evaluation - exact matches from CLI
    env_type: Optional[str] = None  # --env.type (aloha, pusht, xarm, gym_manipulator, hil)
    env_task: Optional[str] = None  # --env.task
    eval_n_episodes: int = 10  # --eval.n_episodes
    eval_batch_size: int = 50  # --eval.batch_size
    eval_use_async_envs: bool = False  # --eval.use_async_envs
    
    # Policy-specific parameters that are commonly used
    policy_device: Optional[str] = "cuda"  # --policy.device
    policy_use_amp: bool = False  # --policy.use_amp
    
    # Optimizer parameters - exact matches from CLI
    optimizer_type: Optional[str] = "adam"  # --optimizer.type (adam, adamw, sgd, multi_adam)
    optimizer_lr: Optional[float] = None  # --optimizer.lr (will use policy default if not set)
    optimizer_weight_decay: Optional[float] = None  # --optimizer.weight_decay
    optimizer_grad_clip_norm: Optional[float] = None  # --optimizer.grad_clip_norm
    
    # Advanced configuration
    use_policy_training_preset: bool = True  # --use_policy_training_preset
    config_path: Optional[str] = None  # --config_path

class TrainingStatus(BaseModel):
    training_active: bool = False
    current_step: int = 0
    total_steps: int = 0
    current_loss: Optional[float] = None
    current_lr: Optional[float] = None
    grad_norm: Optional[float] = None
    epoch_time: Optional[float] = None
    eta_seconds: Optional[float] = None
    available_controls: Dict[str, bool] = {
        "stop_training": False,
        "pause_training": False,
        "resume_training": False
    }

class TrainingManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status = TrainingStatus()
        self.log_queue = queue.Queue()
        self.log_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
    def start_training(self, request: TrainingRequest) -> Dict[str, Any]:
        """Start a training session"""
        if self.status.training_active:
            return {"success": False, "message": "Training is already active"}
        
        try:
            # Create output directory
            output_path = Path(request.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Build the training command
            cmd = self._build_training_command(request)
            logger.info(f"Starting training with command: {' '.join(cmd)}")
            
            # PYTHONUNBUFFERED makes the child's stdout flush per line. Without
            # it, Python block-buffers when stdout is a pipe and our log
            # parser sees nothing until ~4-8KB of output has piled up.
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"

            # Start the training process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=child_env
            )

            # Update status. Reset stale metrics from any prior run so the
            # Monitoring tab doesn't briefly show last run's values.
            self.status.training_active = True
            self.status.total_steps = request.steps
            self.status.current_step = 0
            self.status.current_loss = None
            self.status.current_lr = None
            self.status.grad_norm = None
            self.status.epoch_time = None
            self.status.eta_seconds = None
            self.status.available_controls = {
                "stop_training": True,
                "pause_training": False,  # Not implemented in LeRobot
                "resume_training": False
            }
            
            # Start monitoring threads
            self._start_monitoring()
            
            return {"success": True, "message": "Training started successfully"}
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return {"success": False, "message": f"Failed to start training: {str(e)}"}
    
    def stop_training(self) -> Dict[str, Any]:
        """Stop the current training session"""
        if not self.status.training_active:
            return {"success": False, "message": "No training session is active"}
        
        try:
            if self.process and self.process.poll() is None:
                # Try graceful shutdown first
                self.process.terminate()
                
                # Wait for a bit, then force kill if necessary
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Training process didn't terminate gracefully, forcing kill")
                    self.process.kill()
                    self.process.wait()
            
            self._stop_monitoring_threads()
            self._finalize_status()
            
            return {"success": True, "message": "Training stopped successfully"}
            
        except Exception as e:
            logger.error(f"Failed to stop training: {e}")
            return {"success": False, "message": f"Failed to stop training: {str(e)}"}
    
    def get_status(self) -> TrainingStatus:
        """Get current training status"""
        # Check if process is still running
        if self.process and self.process.poll() is not None:
            # Process has ended
            if self.status.training_active:
                self._stop_monitoring_threads()
                self._finalize_status()
        
        return self.status
    
    def get_logs(self) -> list:
        """Get recent training logs"""
        logs = []
        try:
            while not self.log_queue.empty():
                logs.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        return logs
    
    def _build_training_command(self, request: TrainingRequest) -> list:
        """Build the training command from the request parameters - only using actual CLI parameters"""
        cmd = ["python", "-m", "lerobot.scripts.lerobot_train"]
        
        # Dataset configuration
        cmd.extend(["--dataset.repo_id", request.dataset_repo_id])
        if request.dataset_revision:
            cmd.extend(["--dataset.revision", request.dataset_revision])
        if request.dataset_root:
            cmd.extend(["--dataset.root", request.dataset_root])
        if request.dataset_episodes:
            cmd.extend(["--dataset.episodes"] + [str(ep) for ep in request.dataset_episodes])
        
        # Policy type
        cmd.extend(["--policy.type", request.policy_type])
        
        # Core training parameters
        cmd.extend(["--steps", str(request.steps)])
        cmd.extend(["--batch_size", str(request.batch_size)])
        cmd.extend(["--num_workers", str(request.num_workers)])

        if request.seed is not None:
            cmd.extend(["--seed", str(request.seed)])

        # Policy device and AMP
        if request.policy_device:
            cmd.extend(["--policy.device", request.policy_device])
        cmd.extend(["--policy.use_amp", "true" if request.policy_use_amp else "false"])
        # LeRobot defaults push_to_hub=True and then demands --policy.repo_id.
        # Keep training local by default; uploading is a deliberate action.
        cmd.extend(["--policy.push_to_hub", "false"])

        # Logging and checkpointing
        cmd.extend(["--log_freq", str(request.log_freq)])
        cmd.extend(["--save_freq", str(request.save_freq)])
        cmd.extend(["--eval_freq", str(request.eval_freq)])
        cmd.extend(["--save_checkpoint", "true" if request.save_checkpoint else "false"])

        # Output configuration. Auto-generate a unique sub-path under
        # outputs/train/ when the request carries the default value, so two
        # runs in a row don't collide on LeRobot's "directory exists" guard.
        output_dir = request.output_dir
        if not output_dir or output_dir == DEFAULT_OUTPUT_DIR:
            output_dir = _generate_output_dir(request.policy_type, request.dataset_repo_id)
        cmd.extend(["--output_dir", output_dir])
        cmd.extend(["--resume", "true" if request.resume else "false"])
        if request.job_name:
            cmd.extend(["--job_name", request.job_name])

        # Weights & Biases
        cmd.extend(["--wandb.enable", "true" if request.wandb_enable else "false"])
        if request.wandb_enable:
            if request.wandb_project:
                cmd.extend(["--wandb.project", request.wandb_project])
            if request.wandb_entity:
                cmd.extend(["--wandb.entity", request.wandb_entity])
            if request.wandb_notes:
                cmd.extend(["--wandb.notes", request.wandb_notes])
            if request.wandb_run_id:
                cmd.extend(["--wandb.run_id", request.wandb_run_id])
            if request.wandb_mode:
                cmd.extend(["--wandb.mode", request.wandb_mode])
            cmd.extend(["--wandb.disable_artifact", "true" if request.wandb_disable_artifact else "false"])

        # Environment configuration
        if request.env_type:
            cmd.extend(["--env.type", request.env_type])
        if request.env_task:
            cmd.extend(["--env.task", request.env_task])

        # Evaluation configuration
        cmd.extend(["--eval.n_episodes", str(request.eval_n_episodes)])
        cmd.extend(["--eval.batch_size", str(request.eval_batch_size)])
        cmd.extend(["--eval.use_async_envs", "true" if request.eval_use_async_envs else "false"])

        # Optimizer configuration
        if request.optimizer_type:
            cmd.extend(["--optimizer.type", request.optimizer_type])
        if request.optimizer_lr is not None:
            cmd.extend(["--optimizer.lr", str(request.optimizer_lr)])
        if request.optimizer_weight_decay is not None:
            cmd.extend(["--optimizer.weight_decay", str(request.optimizer_weight_decay)])
        if request.optimizer_grad_clip_norm is not None:
            cmd.extend(["--optimizer.grad_clip_norm", str(request.optimizer_grad_clip_norm)])

        # Advanced options
        cmd.extend(["--use_policy_training_preset", "true" if request.use_policy_training_preset else "false"])
        if request.config_path:
            cmd.extend(["--config_path", request.config_path])

        return cmd
    
    def _start_monitoring(self):
        """Start monitoring threads for process output and status"""
        self._stop_monitoring.clear()
        
        # Start log monitoring thread
        self.log_thread = threading.Thread(
            target=self._monitor_logs,
            daemon=True
        )
        self.log_thread.start()
        
        # Start status monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_status,
            daemon=True
        )
        self.monitor_thread.start()
    
    def _stop_monitoring_threads(self):
        """Stop all monitoring threads"""
        self._stop_monitoring.set()
        
        if self.log_thread and self.log_thread.is_alive():
            self.log_thread.join(timeout=2)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
    
    def _monitor_logs(self):
        """Monitor training process logs"""
        if not self.process:
            return
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if self._stop_monitoring.is_set():
                    break
                
                if line.strip():
                    # Parse training information from log line
                    self._parse_log_line(line.strip())
                    
                    # Add to log queue (keep last 1000 lines)
                    if self.log_queue.qsize() >= 1000:
                        try:
                            self.log_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.log_queue.put({
                        "timestamp": time.time(),
                        "message": line.strip()
                    })
        
        except Exception as e:
            logger.error(f"Error monitoring logs: {e}")
    
    def _monitor_status(self):
        """Monitor training process status"""
        while not self._stop_monitoring.is_set() and self.process:
            try:
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process has ended
                    break
                
                # Update available controls
                self.status.available_controls["stop_training"] = True
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error monitoring status: {e}")
                break
    
    def _parse_log_line(self, line: str):
        """Pull current_step / total_steps / ETA / loss / lr / grad_norm from
        whichever LeRobot output line we just received.

        Two complementary sources:
          * The tqdm progress bar (`Training:  1%|...| 125/10000 [02:02<2:36:10, 1.05step/s]`)
            fires roughly once per second and gives current_step + total_steps + ETA.
            Without this the Progress card stays at 0 until the first
            log_freq tick — and with the default log_freq=250, that's a
            very long time on a slow device.
          * The per-log_freq metric line
            (`INFO ... step:N smpl:... loss:X grdn:Y lr:Z ...`)
            is the only place loss / lr / grad_norm appear.
        """
        try:
            tqdm_match = _TQDM_RE.search(line)
            if tqdm_match:
                try:
                    self.status.current_step = int(tqdm_match.group(1))
                    total = int(tqdm_match.group(2))
                    if total > 0:
                        self.status.total_steps = total
                    eta = _parse_duration(tqdm_match.group(3))
                    if eta is not None:
                        self.status.eta_seconds = eta
                except (ValueError, IndexError):
                    pass

            # log_freq metric line — has loss/lr/grdn alongside step.
            if "step:" in line and "loss:" in line:
                if "step:" in line:
                    step_part = line.split("step:")[1].split()[0]
                    try:
                        self.status.current_step = int(step_part.replace(",", ""))
                    except ValueError:
                        pass
                if "loss:" in line:
                    try:
                        self.status.current_loss = float(line.split("loss:")[1].split()[0])
                    except ValueError:
                        pass
                if "lr:" in line:
                    try:
                        self.status.current_lr = float(line.split("lr:")[1].split()[0])
                    except ValueError:
                        pass
                if "grdn:" in line:
                    try:
                        self.status.grad_norm = float(line.split("grdn:")[1].split()[0])
                    except ValueError:
                        pass

        except Exception as e:
            logger.debug(f"Error parsing log line '{line}': {e}")
    
    def _finalize_status(self):
        """Mark a run as ended without wiping its final metrics.

        Keep current_step, total_steps, current_loss, current_lr, grad_norm,
        eta_seconds so the Monitoring tab can show the last values from the
        completed (or stopped) run. The next start_training call resets the
        stale fields explicitly.
        """
        self.status.training_active = False
        self.status.available_controls = {
            "stop_training": False,
            "pause_training": False,
            "resume_training": False,
        }
        self.process = None

# Global training manager instance
training_manager = TrainingManager()

# Handler functions for FastAPI endpoints
def handle_start_training(request: TrainingRequest) -> Dict[str, Any]:
    """Handle start training request"""
    return training_manager.start_training(request)

def handle_stop_training() -> Dict[str, Any]:
    """Handle stop training request"""
    return training_manager.stop_training()

def handle_training_status() -> TrainingStatus:
    """Handle training status request"""
    return training_manager.get_status()

def handle_training_logs() -> Dict[str, Any]:
    """Handle training logs request"""
    logs = training_manager.get_logs()
    return {"logs": logs} 
