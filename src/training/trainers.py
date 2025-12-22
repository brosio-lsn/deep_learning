import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Any
import torch
from tqdm import tqdm


class BlackboardTrainer:
    def __init__(
        self,
        *,
        model,
        optimizer,
        device,
        train_loader,
        val_loader,
        train_cfg,
        board_cfg,
        accuracy_with_splits,
        masked_cross_entropy_fn,  
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_cfg = train_cfg
        self.board_cfg = board_cfg
        self.acc  = accuracy_with_splits
        self.loss = masked_cross_entropy_fn

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_carry_acc": [],
            "train_digit_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_carry_acc": [],
            "val_digit_acc": [],
        }

        self.run_dir = os.path.join(self.train_cfg.out_dir, self.train_cfg.exp_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.global_step = 0
    

    def _save_checkpoint(self):
        if not (self.cfg.enable_docs and self.cfg.save_model):
            return
        path = os.path.join(self.run_dir,"")
        torch.save({"model_state_dict": self.model.state_dict()}, path)

    def _save_history_json(self):
        if not self.cfg.enable_docs:
            return
        path = os.path.join(self.run_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _plot_history(self):
        if not (self.cfg.enable_docs and self.cfg.enable_plots):
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("[WARN] matplotlib not available; skipping plots.")
            return

        def plot_curve(train_key, val_key, title, ylabel, filename):
            plt.figure()
            plt.plot(self.history[train_key], label="train")
            plt.plot(self.history[val_key], label="val")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, filename))
            plt.close()

        plot_curve("train_loss", "val_loss", "Loss per token", "Cross Entropy", "loss.png")
        plot_curve("train_acc", "val_acc", "Accuracy (masked)", "Accuracy", "acc.png")
        plot_curve("train_carry_acc", "val_carry_acc", "Carry accuracy", "Accuracy", "carry_acc.png")
        plot_curve("train_digit_acc", "val_digit_acc", "Digit accuracy", "Accuracy", "digit_acc.png")
    
    def _run_epoch(self, *, loader, mode: str, epoch: int) -> Dict[str, float]:
   
        is_train = (mode == "train")
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        total_carry_correct = 0
        total_carry_tokens = 0
        total_digit_correct = 0
        total_digit_tokens = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{self.cfg.num_epochs} [{mode}]")

        with torch.set_grad_enabled(is_train):
            for i, batch in enumerate(pbar):

                input_ids  = batch["input_ids"].to(device) 
                target_ids = batch["target_ids"].to(device)  
                mask       = batch["mask"].to(device)       

                if is_train:
                    self.optimizer.zero_grad()

                logits, _ = self.model(input_ids)
                loss = self.loss(logits, target_ids, loss_mask)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                batch_tokens = mask.sum().item()
                batch_loss = loss.item()

                (b_total_correct,
                b_total_tokens,
                b_carry_correct,
                b_carry_tokens,
                b_digit_correct,
                b_digit_tokens) = self.acc(
                    logits, target_ids, mask, H=self.board_cfg.H, W=self.board_cfg.W
                )

                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
                total_correct += b_total_correct
                total_carry_correct += b_carry_correct
                total_carry_tokens += b_carry_tokens
                total_digit_correct += b_digit_correct
                total_digit_tokens += b_digit_tokens

                batch_acc = b_total_correct / max(b_total_tokens, 1)
                pbar.set_postfix(loss=batch_loss, acc=batch_acc)

            avg_loss = total_loss / max(total_tokens, 1)
            avg_acc  = total_correct / max(total_tokens, 1)
            carry_acc = (
                total_carry_correct / total_carry_tokens
                if total_carry_tokens > 0 else 0.0
            )
            digit_acc = (
                total_digit_correct / total_digit_tokens
                if total_digit_tokens > 0 else 0.0
            )

        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "carry_acc": carry_acc,
            "digit_acc": digit_acc,
        }

     def fit(self):
        print("Starting training CoT transformer ...")
        for epoch in range(1, self.cfg.num_epochs + 1):
            train_metrics = self._run_epoch(loader=self.train_loader, mode="train", epoch=epoch)
            val_metrics = self._run_epoch(loader=self.val_loader, mode="val", epoch=epoch)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["train_carry_acc"].append(train_metrics["carry_acc"])
            self.history["train_digit_acc"].append(train_metrics["digit_acc"])

            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["val_carry_acc"].append(val_metrics["carry_acc"])
            self.history["val_digit_acc"].append(val_metrics["digit_acc"])

            # print epoch summary (same content)
            print(
                f"\nEpoch {epoch}/{self.cfg.num_epochs} "
                f"| train loss/token: {train_metrics['loss']:.4f} "
                f"| train acc(masked): {train_metrics['acc']:.4f} "
                f"| train carry acc: {train_metrics['carry_acc']:.4f} "
                f"| train digit acc: {train_metrics['digit_acc']:.4f}"
            )
            print(
                f"Epoch {epoch}/{self.cfg.num_epochs} "
                f"| val   loss/token: {val_metrics['loss']:.4f} "
                f"| val   acc(masked): {val_metrics['acc']:.4f} "
                f"| val   carry acc: {val_metrics['carry_acc']:.4f} "
                f"| val   digit acc: {val_metrics['digit_acc']:.4f}"
            )
            print("-" * 80)
           
           
        self._save_checkpoint()
        self._plot_history()
        self._save_history_json()

        # optional: print the whole history (kept)
        print("Training history:")
        for e in range(self.cfg.num_epochs):
            print(
                f"Epoch {e+1}: "
                f"train_loss={self.history['train_loss'][e]:.4f}, "
                f"train_acc={self.history['train_acc'][e]:.4f}, "
                f"train_carry_acc={self.history['train_carry_acc'][e]:.4f}, "
                f"train_digit_acc={self.history['train_digit_acc'][e]:.4f}, "
                f"val_loss={self.history['val_loss'][e]:.4f}, "
                f"val_acc={self.history['val_acc'][e]:.4f}, "
                f"val_carry_acc={self.history['val_carry_acc'][e]:.4f}, "
                f"val_digit_acc={self.history['val_digit_acc'][e]:.4f}"
            )


class COTTrainer:

    def __init__(
        self,
        *,
        model,
        optimizer,
        device,
        train_loader,
        val_loader,
        id2tok,
        train_cfg,
        masked_cross_entropy_fn,  
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.id2tok = id2tok
        self.cfg = train_cfg
        self.masked_ce = masked_cross_entropy_fn

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_carry_acc": [],
            "train_digit_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_carry_acc": [],
            "val_digit_acc": [],
        }


        self.run_dir = os.path.join(self.cfg.out_dir, self.cfg.exp_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.global_step = 0
    

    def _print_tokens(self, string: str, ids: torch.Tensor, id2tok: Dict[int, str]):
        ids = ids.detach().cpu()
        s = [id2tok[i.item()] for i in ids]
        print(f"{string} {s}")

    def _print_batch_debug(
        self,
        mode: str,
        epoch: int,
        i: int,
        num_batches: int,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        pred_ids: torch.Tensor,
        digit_pos: int,
        carry_pos: int,
        id2tok: Dict[int, str],
        log_interval: float,
    ):
        interval = max(int(num_batches * log_interval), 1)
        if (i + 1) % interval != 0:
            return

        header = f"{mode} / Epoch: {epoch} / Batch: {i+1}"
        print(f"{header}\n")
        print("-" * 100)
        self.print_tokens("Input IDS of first element of the batch: ", input_ids[0], id2tok)
        self.print_tokens("Target IDS of first element of the batch: ", target_ids[0], id2tok)
        self.print_tokens("Predicted IDS of first element of the batch: ", pred_ids[0], id2tok)

        print("Digit:")
        print("Pred: ", id2tok[pred_ids[0][digit_pos].item()])
        print("Target: ", id2tok[target_ids[0][digit_pos].item()])

        print("Carry:")
        print("Pred: ", id2tok[pred_ids[0][carry_pos].item()])
        print("Target: ", id2tok[target_ids[0][carry_pos].item()])
        print("-" * 100)


    def _save_checkpoint(self):
        if not (self.cfg.enable_docs and self.cfg.save_model):
            return
        path = os.path.join(self.run_dir,"")
        torch.save({"model_state_dict": self.model.state_dict()}, path)

    def _save_history_json(self):
        if not self.cfg.enable_docs:
            return
        path = os.path.join(self.run_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _plot_history(self):
        if not (self.cfg.enable_docs and self.cfg.enable_plots):
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("[WARN] matplotlib not available; skipping plots.")
            return

        def plot_curve(train_key, val_key, title, ylabel, filename):
            plt.figure()
            plt.plot(self.history[train_key], label="train")
            plt.plot(self.history[val_key], label="val")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, filename))
            plt.close()

        plot_curve("train_loss", "val_loss", "Loss per token", "Cross Entropy", "loss.png")
        plot_curve("train_acc", "val_acc", "Accuracy (masked)", "Accuracy", "acc.png")
        plot_curve("train_carry_acc", "val_carry_acc", "Carry accuracy", "Accuracy", "carry_acc.png")
        plot_curve("train_digit_acc", "val_digit_acc", "Digit accuracy", "Accuracy", "digit_acc.png")

    def _run_epoch(self, *, loader, mode: str, epoch: int) -> Dict[str, float]:
   
        is_train = (mode == "train")
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        total_carry_correct = 0
        total_carry_tokens = 0
        total_digit_correct = 0
        total_digit_tokens = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{self.cfg.num_epochs} [{mode}]")

        with torch.set_grad_enabled(is_train):
            for i, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["label_ids"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                attn_mask = batch["attn_mask"][0].to(self.device)
                digit_positions = batch["digit_pos"].to(self.device)
                carry_positions = batch["carry_pos"].to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                logits, _ = self.model(input_ids, src_mask=attn_mask)
                loss = self.masked_ce(logits, target_ids, loss_mask)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                batch_tokens = loss_mask.sum().item()
                batch_loss = loss.item()

                pred_ids = logits.argmax(dim=-1)
                correct = (pred_ids == target_ids)

                b_total_correct = (correct & loss_mask).sum().item()

                digit_pos = digit_positions[0].item()
                carry_pos = carry_positions[0].item()

                digit_mask = torch.zeros_like(loss_mask).bool()
                digit_mask[:, digit_pos] = True

                carry_mask = torch.zeros_like(loss_mask).bool()
                carry_mask[:, carry_pos] = True

                b_digit_correct = (correct & digit_mask).sum().item()
                b_digit_tokens = digit_mask.sum().item()

                b_carry_correct = (correct & carry_mask).sum().item()
                b_carry_tokens = carry_mask.sum().item()

                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
                total_correct += b_total_correct
                total_carry_correct += b_carry_correct
                total_carry_tokens += b_carry_tokens
                total_digit_correct += b_digit_correct
                total_digit_tokens += b_digit_tokens

                batch_acc = b_total_correct / max(batch_tokens, 1)
                pbar.set_postfix(loss=batch_loss, acc=batch_acc)

                # Print results for first batch of element of the batch if condition met
                self.print_batch_debug(
                        mode="Training" if is_train else "Validation"
                        epoch=epoch,
                        i=i,
                        num_batches=len(loader),
                        input_ids=input_ids,
                        target_ids=target_ids,
                        pred_ids=pred_ids,
                        digit_pos=digit_pos,
                        carry_pos=carry_pos,
                        id2tok=self.id2tok,
                        log_interval=self.cfg.log_interval,
                )

        avg_loss = total_loss / max(total_tokens, 1)
        avg_acc = total_correct / max(total_tokens, 1)
        carry_acc = (total_carry_correct / total_carry_tokens) if total_carry_tokens > 0 else 0.0
        digit_acc = (total_digit_correct / total_digit_tokens) if total_digit_tokens > 0 else 0.0

        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "carry_acc": carry_acc,
            "digit_acc": digit_acc,
        }

    def fit(self):
        print("Starting training CoT transformer ...")
        for epoch in range(1, self.cfg.num_epochs + 1):
            train_metrics = self._run_epoch(loader=self.train_loader, mode="train", epoch=epoch)
            val_metrics = self._run_epoch(loader=self.val_loader, mode="val", epoch=epoch)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["train_carry_acc"].append(train_metrics["carry_acc"])
            self.history["train_digit_acc"].append(train_metrics["digit_acc"])

            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["val_carry_acc"].append(val_metrics["carry_acc"])
            self.history["val_digit_acc"].append(val_metrics["digit_acc"])


            # print epoch summary (same content)
            print(
                f"\nEpoch {epoch}/{self.cfg.num_epochs} "
                f"| train loss/token: {train_metrics['loss']:.4f} "
                f"| train acc(masked): {train_metrics['acc']:.4f} "
                f"| train carry acc: {train_metrics['carry_acc']:.4f} "
                f"| train digit acc: {train_metrics['digit_acc']:.4f}"
            )
            print(
                f"Epoch {epoch}/{self.cfg.num_epochs} "
                f"| val   loss/token: {val_metrics['loss']:.4f} "
                f"| val   acc(masked): {val_metrics['acc']:.4f} "
                f"| val   carry acc: {val_metrics['carry_acc']:.4f} "
                f"| val   digit acc: {val_metrics['digit_acc']:.4f}"
            )
            print("-" * 80)


        self._save_checkpoint()
        self._plot_history()
        self._save_history_json()

        # optional: print the whole history (kept)
        print("Training history:")
        for e in range(self.cfg.num_epochs):
            print(
                f"Epoch {e+1}: "
                f"train_loss={self.history['train_loss'][e]:.4f}, "
                f"train_acc={self.history['train_acc'][e]:.4f}, "
                f"train_carry_acc={self.history['train_carry_acc'][e]:.4f}, "
                f"train_digit_acc={self.history['train_digit_acc'][e]:.4f}, "
                f"val_loss={self.history['val_loss'][e]:.4f}, "
                f"val_acc={self.history['val_acc'][e]:.4f}, "
                f"val_carry_acc={self.history['val_carry_acc'][e]:.4f}, "
                f"val_digit_acc={self.history['val_digit_acc'][e]:.4f}"
            )
