"""
PatchTST AI Model for Crypto Price Prediction
Uses Transformer architecture optimized for time series forecasting
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import pickle
import psutil


class PatchTST(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words
    Simplified implementation for crypto price prediction
    """

    def __init__(
        self,
        seq_len: int = 60,   # Input sequence length
        pred_len: int = 5,   # Prediction length (we use last step as 15‑min change target)
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_features: int = 10,  # Number of features (technical indicators)
    ):
        super(PatchTST, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_features = n_features

        # Patch embedding
        self.patch_len = 16
        self.stride = 8
        self.patch_num = int((seq_len - self.patch_len) / self.stride + 1)

        # Input projection
        self.input_projection = nn.Linear(self.patch_len, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.patch_num, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers,
        )

        # Output projection
        self.output_projection = nn.Linear(d_model * self.patch_num, pred_len)

        # Feature-wise output (for multivariate)
        self.feature_projection = nn.Linear(n_features, 1)

        self.dropout = nn.Dropout(dropout)

    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Create patches from input sequence (pure tensor ops)."""
        batch_size, seq_len, n_features = x.shape

        patch_starts = torch.arange(
            0, self.patch_num, device=x.device, dtype=torch.long
        ) * self.stride
        patch_offsets = torch.arange(
            self.patch_len, device=x.device, dtype=torch.long
        )
        patch_indices = patch_starts.unsqueeze(1) + patch_offsets.unsqueeze(0)

        patch_indices_expanded = patch_indices.unsqueeze(0)
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).unsqueeze(2)
        patch_indices_batch = patch_indices_expanded.expand(batch_size, -1, -1)

        feature_indices = torch.arange(n_features, device=x.device).view(
            1, 1, 1, -1
        )

        batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1, 1)
        seq_idx = patch_indices_batch.unsqueeze(-1)

        patches = x[batch_idx.squeeze(-1), seq_idx.squeeze(-1), :]
        patches = patches.reshape(batch_size, self.patch_num, -1)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            [batch, pred_len]
        """
        batch_size = x.shape[0]

        patches = self.create_patches(x)

        patches_reshaped = patches.reshape(
            batch_size, self.patch_num, self.patch_len, self.n_features
        )
        patch_avg = patches_reshaped.mean(dim=-1)
        patch_avg_flat = patch_avg.reshape(-1, self.patch_len)
        patch_proj_flat = self.input_projection(patch_avg_flat)
        x = patch_proj_flat.reshape(batch_size, self.patch_num, self.d_model)

        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, -1)
        output = self.output_projection(x)
        return output


class CryptoDataset(Dataset):
    """Dataset for crypto training data"""

    def __init__(self, training_data: List[Tuple]):
        """
        training_data: List of (features_sequence, target_price_change) tuples
        """
        self.data = []

        for features_seq, target in training_data:
            if len(features_seq) >= 60:
                seq = features_seq[-60:]
                try:
                    feature_valid = True
                    for feature in seq:
                        if isinstance(feature, (list, np.ndarray)):
                            for val in feature:
                                if not isinstance(val, (int, float)) or not np.isfinite(
                                    float(val)
                                ):
                                    feature_valid = False
                                    break
                        elif not isinstance(feature, (int, float)) or not np.isfinite(
                            float(feature)
                        ):
                            feature_valid = False
                        if not feature_valid:
                            break
                    if not feature_valid:
                        continue
                except (TypeError, ValueError):
                    continue

                try:
                    target_float = float(target) if target is not None else None
                    if target_float is None or not np.isfinite(target_float):
                        continue
                except (TypeError, ValueError):
                    continue

                self.data.append((seq, target_float))

        if len(self.data) > 0:
            X_list, y_list = zip(*self.data)
            self.X = np.array(X_list, dtype=np.float32)
            self.y = np.array(y_list, dtype=np.float32)

            if np.any(~np.isfinite(self.X)):
                self.X = np.nan_to_num(self.X, nan=0.0, posinf=1.0, neginf=-1.0)
            if np.any(~np.isfinite(self.y)):
                self.y = np.nan_to_num(self.y, nan=0.0, posinf=0.0, neginf=0.0)

            y_clipped = np.clip(self.y, -50.0, 50.0)
            self.y_normalized = y_clipped / 10.0
        else:
            self.X = np.array([], dtype=np.float32).reshape(0, 60, 10)
            self.y_normalized = np.array([], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(
            [self.y_normalized[idx]]
        )


class AIPredictor:
    """AI-based price predictor using PatchTST"""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or "models/patchtst_crypto.pth"
        self.scaler = None
        self.is_trained = False

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        os.makedirs("models", exist_ok=True)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize PatchTST model"""
        self.model = PatchTST(
            seq_len=60,
            pred_len=5,  # We only use the last element as 15‑min change target
            d_model=128,
            n_heads=8,
            e_layers=3,
            d_ff=256,
            dropout=0.1,
            n_features=10,
        )

        self.model = self.model.to(self.device)

        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                self.is_trained = True
            except Exception:
                self.is_trained = False
        else:
            self.is_trained = False

    def prepare_features(self, indicators: Dict) -> np.ndarray:
        """Prepare features from technical indicators"""
        features = [
            indicators.get("velocity", 0.0),
            indicators.get("momentum", 0.0),
            indicators.get("rsi", 50.0) / 100.0,
            indicators.get("ema_position_score", 0.0),
            indicators.get("trend_strength", 0.5),
            indicators.get("macd_histogram", 0.0),
            indicators.get("stochastic_k", 50.0) / 100.0,
            indicators.get("bollinger_position", 0.5),
            indicators.get("atr_normalized", 0.0),
            indicators.get("adx", 0.0) / 100.0,
        ]
        return np.array(features, dtype=np.float32)

    def predict(self, feature_history: List[np.ndarray], current_price: float) -> Dict:
        """Predict future price using AI model"""
        if not self.is_trained or len(feature_history) < 60:
            return {
                "predicted_price": current_price,
                "predicted_change_pct": 0.0,
                "confidence": 0.3,
                "method": "fallback",
            }

        try:
            seq_len = min(60, len(feature_history))
            features_seq = feature_history[-seq_len:]

            if len(features_seq) < 60:
                padding = np.zeros((60 - len(features_seq), 10), dtype=np.float32)
                features_seq = np.concatenate([padding, features_seq], axis=0)

            if isinstance(features_seq, list):
                features_seq = np.array(features_seq, dtype=np.float32)

            input_tensor = (
                torch.FloatTensor(features_seq).unsqueeze(0).to(self.device)
            )

            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_tensor)
                predicted_change_normalized = prediction[0, -1].item()

            predicted_change_pct = predicted_change_normalized * 10.0
            predicted_price = current_price * (1 + predicted_change_pct / 100)
            confidence = min(0.9, 0.5 + abs(predicted_change_normalized) * 2)

            return {
                "predicted_price": predicted_price,
                "predicted_change_pct": predicted_change_pct,
                "confidence": confidence,
                "method": "ai_patchtst",
            }

        except Exception:
            return {
                "predicted_price": current_price,
                "predicted_change_pct": 0.0,
                "confidence": 0.3,
                "method": "error_fallback",
            }

    def save_model(self, path: Optional[str] = None):
        """Save trained model"""
        save_path = path or self.model_path
        torch.save(self.model.state_dict(), save_path)

    def export_to_coreml(self, export_path: str = "models/patchtst_coreml.mlmodel"):
        """
        Export trained model to Core ML format.
        Silent; returns True/False.
        """
        try:
            import coremltools as ct
        except ImportError:
            return False

        if not self.is_trained and not os.path.exists(self.model_path):
            return False

        if not hasattr(self, "model") or self.model is None:
            self._initialize_model()

        original_device = self.device
        self.model = self.model.cpu()
        self.model.eval()

        example_input = torch.rand(1, 60, 10)

        try:
            self.model.eval()
            with torch.no_grad():
                example_inputs = [
                    torch.rand(1, 60, 10),
                    torch.rand(1, 60, 10),
                    torch.rand(1, 60, 10),
                ]

                scripted_model = torch.jit.trace(
                    self.model, example_inputs[0], strict=False
                )
                scripted_model.eval()
                for test_input in example_inputs:
                    _ = scripted_model(test_input)

            try:
                coreml_model = ct.convert(
                    scripted_model,
                    inputs=[ct.TensorType(shape=example_input.shape, name="features")],
                    outputs=[ct.TensorType(name="prediction")],
                    compute_units=ct.ComputeUnit.ALL,
                    minimum_deployment_target=ct.target.macOS13,
                )
            except Exception as direct_error:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                    onnx_path = tmp.name

                torch.onnx.export(
                    scripted_model,
                    example_input,
                    onnx_path,
                    input_names=["features"],
                    output_names=["prediction"],
                    dynamic_axes={
                        "features": {0: "batch"},
                        "prediction": {0: "batch"},
                    },
                    opset_version=14,
                )

                coreml_model = ct.convert(
                    onnx_path,
                    inputs=[ct.TensorType(shape=example_input.shape, name="features")],
                    outputs=[ct.TensorType(name="prediction")],
                    compute_units=ct.ComputeUnit.ALL,
                    minimum_deployment_target=ct.target.macOS13,
                )

                os.unlink(onnx_path)

            os.makedirs(
                os.path.dirname(export_path) if os.path.dirname(export_path) else ".",
                exist_ok=True,
            )
            coreml_model.save(export_path)

            self.model = self.model.to(original_device)
            return True

        except Exception:
            if hasattr(self, "model") and self.model is not None:
                self.model = self.model.to(original_device)
            return False

    def train_model(
        self,
        training_data: List[Tuple],
        epochs: int = 50,
        start_epoch: int = 0,
        max_samples: int = None,
    ):
        """
        Train the model on historical data using MPS (Mac GPU) with DataLoader.
        Target is interpreted as 15‑minute percent change.
        """
        if len(training_data) < 100:
            return

        if max_samples is None:
            if len(training_data) > 100000:
                max_samples = 100000

        if max_samples and len(training_data) > max_samples:
            training_data = training_data[:max_samples]

        dataset = CryptoDataset(training_data)

        if len(dataset) < 50:
            return

        batch_size = 4
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5
        )

        if start_epoch == 0 and self.is_trained:
            checkpoint_files = []
            if os.path.exists("models"):
                for f in os.listdir("models"):
                    if f.startswith("patchtst_checkpoint_epoch") and f.endswith(".pth"):
                        try:
                            epoch_num = int(
                                f.replace("patchtst_checkpoint_epoch", "").replace(
                                    ".pth", ""
                                )
                            )
                            checkpoint_files.append((epoch_num, f))
                        except Exception:
                            pass

            if checkpoint_files:
                last_checkpoint_epoch = max([e for e, _ in checkpoint_files])
                if last_checkpoint_epoch < epochs:
                    start_epoch = last_checkpoint_epoch
                    checkpoint_path = (
                        f"models/patchtst_checkpoint_epoch{last_checkpoint_epoch}.pth"
                    )
                    try:
                        self.model.load_state_dict(
                            torch.load(checkpoint_path, map_location=self.device)
                        )
                    except Exception:
                        start_epoch = 0

        self.model.train()
        best_loss = float("inf")

        import time as _time

        print(f"\n=== Training PatchTST model for {epochs} epochs on {len(dataset)} samples ===")
        print(f"Device: {self.device.type.upper()}, batch_size=4\n")

        for epoch in range(start_epoch, epochs):
            epoch_start_time = _time.time()
            total_loss = 0
            num_batches = 0

            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions[:, -1:], batch_y)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                if self.device.type == "mps":
                    torch.mps.synchronize()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 1000 == 0 and batch_idx > 0:
                    import gc

                    gc.collect()
                    if self.device.type == "mps" and hasattr(
                        torch.mps, "empty_cache"
                    ):
                        torch.mps.empty_cache()

            if num_batches == 0:
                print(f"Epoch {epoch+1}: no valid batches, stopping.")
                break

            avg_loss = total_loss / num_batches

            if not np.isfinite(avg_loss):
                print(f"Epoch {epoch+1}: loss is NaN/Inf, stopping.")
                break

            scheduler.step(avg_loss)

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            display_epoch = epoch + 1
            print(f"Epoch {display_epoch}/{epochs} | loss={avg_loss:.6f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.6f} | mem={memory_mb:.0f}MB")

            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"models/patchtst_checkpoint_epoch{epoch+1}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model()

            if self.device.type == "mps":
                import gc

                if (epoch + 1) % 5 == 0:
                    gc.collect()
                    torch.mps.empty_cache() if hasattr(
                        torch.mps, "empty_cache"
                    ) else None

            if (epoch + 1) % 3 == 0:
                _time.sleep(1)
            elif (epoch + 1) % 10 == 0:
                _time.sleep(2)

        self.is_trained = True
        self.save_model()


