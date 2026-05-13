#!/usr/bin/env python3
"""
Hailo Inference Pipeline Module
"""

# code adapted from https://github.com/SangatsuUsagi/hailo_inference_pipeline/tree/main 

import argparse
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatOrder,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)



TIMEOUT_MS: int = 1000


# ============================================================================
# Exception Classes
# ============================================================================


class InferenceError(Exception):
    """Base exception for inference-related errors."""

    pass


class InferenceSubmitError(InferenceError):
    """Exception raised when inference submission fails."""

    pass


class InferenceTimeoutError(InferenceError):
    """Exception raised when inference operation times out."""

    pass


class InferenceWaitError(InferenceError):
    """Exception raised when waiting for inference results fails."""

    pass


class InferencePipelineError(InferenceError):
    """Exception raised during synchronous inference pipeline execution."""

    pass


# ============================================================================
# Helper Functions for Exception Detection
# ============================================================================


def is_hailo_timeout_exception(e: Exception) -> bool:
    """
    Check if an exception is a Hailo timeout exception.

    Args:
        e: Exception to check

    Returns:
        True if exception is HailoRTTimeout
    """
    exception_type = type(e).__name__
    exception_str = str(e).lower()
    return (
        exception_type == "HailoRTTimeout"
        or "timeout" in exception_type.lower()
        or "timeout" in exception_str
        or "timed out" in exception_str
    )


def is_hailo_exception(e: Exception) -> bool:
    """
    Check if an exception is a Hailo runtime exception.

    Args:
        e: Exception to check

    Returns:
        True if exception is HailoRTException or derived type
    """
    exception_type = type(e).__name__
    return (
        exception_type.startswith("HailoRT")
        or "hailo" in exception_type.lower()
        or hasattr(e, "__module__")
        and "hailo" in str(getattr(e, "__module__", "")).lower()
    )


# ============================================================================
# InferPipeline Class
# ============================================================================




class InferPipeline:
    """
    Manages asynchronous and blocking inference pipelines for Hailo models.

    Supports both synchronous and asynchronous execution modes with various
    post-processing capabilities for classification and detection tasks.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __init__(
        self,
        net_path: str,
        batch_size: int
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            net_path: Path to the HEF model file
            batch_size: Number of inputs to process in a single batch

        Raises:
            RuntimeError: If initialization fails
        """
        # Initialize all attributes first (for safe cleanup)
        self.out_results: Dict[str, np.ndarray] = {}

        self.configured_infer_model: Optional[Any] = None
        self.bindings: Optional[Any] = None
        self.job: Optional[Any] = None

        # Initialize device-related attributes as None
        self.vdevice: Optional[VDevice] = None
        self.infer_model: Optional[Any] = None
        self.hef: Optional[HEF] = None
        self.network_group: Optional[Any] = None
        self.input_vstreams_params: Optional[Any] = None
        self.output_vstreams_params: Optional[Any] = None

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        try:
            self.vdevice = VDevice(params)

            self.hef = HEF(net_path)
            configure_params = ConfigureParams.create_from_hef(
                hef=self.hef, interface=HailoStreamInterface.PCIe
            )
            self.network_group = (
               self.vdevice.configure(self.hef, configure_params)
            )[0]

            self.input_vstreams_params = InputVStreamParams.make(
                self.network_group,
                format_type=FormatType.UINT8,
            )
            self.output_vstreams_params = OutputVStreamParams.make(
                self.network_group, format_type=FormatType.FLOAT32
            )

        except Exception as e:
            # Clean up any partially initialized resources
            try:
                if self.vdevice is not None:
                    self.vdevice.release()
            except Exception:
                pass  # Ignore cleanup errors

            print(f"Error during initialize: {e}")
            raise RuntimeError(f"Failed to initialize InferPipeline: {e}") from e


    def close(self) -> None:
        """Clean up allocated resources safely."""
        try:
            if self.configured_infer_model is not None:
                if hasattr(self.configured_infer_model, "release"):
                    self.configured_infer_model.release()
                self.configured_infer_model = None

            if self.network_group is not None:
                self.network_group = None

            if self.vdevice is not None:
                self.vdevice.release()
                self.vdevice = None
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")




    def infer_pipeline(self, infer_inputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform synchronous (blocking) inference on input data.

        Args:
            infer_inputs: List of input data arrays for the model

        Returns:
            Dictionary mapping output layer names to inference results

        Raises:
            InferencePipelineError: If synchronous inference fails
            InferenceTimeoutError: If operation times out
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not infer_inputs:
            raise ValueError("infer_inputs cannot be empty")

        infer_results: Dict[str, np.ndarray] = {}

        try:
            # Prepare input data dictionary
            input_data: Dict[str, np.ndarray] = {}
            input_vstream_infos = self.hef.get_input_vstream_infos()

            if len(infer_inputs) != len(input_vstream_infos):
                raise ValueError(
                    f"Expected {len(input_vstream_infos)} inputs, "
                    f"got {len(infer_inputs)}"
                )

            for i, input_vstream_info in enumerate(input_vstream_infos):
                if infer_inputs[i] is None:
                    raise ValueError(f"Input at index {i} cannot be None")

                # Add batch dimension if needed
                input_data[input_vstream_info.name] = infer_inputs[i][np.newaxis, :]

            # Execute synchronous inference pipeline with Hailo exception handling
            try:
                with InferVStreams(
                    self.network_group,
                    self.input_vstreams_params,
                    self.output_vstreams_params,
                ) as infer_pipeline:
                    # Run inference
                    try:
                        buffer = infer_pipeline.infer(input_data)
                    except Exception as e:
                        if is_hailo_timeout_exception(e):
                            print(f"Inference timeout: {e}")
                            raise InferenceTimeoutError(
                                f"Synchronous inference timed out: {e}"
                            ) from e
                        elif is_hailo_exception(e):
                            print(f"Hailo runtime error during inference: {e}")
                            raise InferencePipelineError(
                                f"Hailo device error during inference: {e}"
                            ) from e
                        else:
                            raise

                    # Extract results
                    output_vstream_infos = self.hef.get_output_vstream_infos()
                    for i, output_vstream_info in enumerate(output_vstream_infos):
                        output_name = output_vstream_info.name

                        if output_name not in buffer:
                            available_outputs = list(buffer.keys())
                            raise InferencePipelineError(
                                f"Expected output '{output_name}' not found in results. "
                                f"Available outputs: {available_outputs}"
                            )

                        try:
                            infer_results[output_name] = buffer[
                                output_name
                            ].squeeze()
                        except Exception as e:
                            print(f"Error processing output '{output_name}': {e}")
                            raise InferencePipelineError(
                                f"Failed to process output '{output_name}': {e}"
                            ) from e

            except (InferencePipelineError, InferenceTimeoutError):
                raise

            except Exception as e:
                if is_hailo_exception(e):
                    print(f"Hailo error in inference pipeline context: {e}")
                    raise InferencePipelineError(
                        f"Hailo device error in inference pipeline: {e}"
                    ) from e
                else:
                    raise

        except (ValueError, InferencePipelineError, InferenceTimeoutError):
            raise

        except AttributeError as e:
            print(f"Initialization error during synchronous inference: {e}")
            raise InferencePipelineError(
                f"Inference pipeline not properly initialized: {e}"
            ) from e

        except KeyError as e:
            print(f"Missing expected data during inference: {e}")
            raise InferencePipelineError(
                f"Inference failed due to missing data: {e}"
            ) from e

        except Exception as e:
            print(f"Unexpected error during synchronous inference: {e}")
            raise InferencePipelineError(
                f"Unexpected error during inference: {e}"
            ) from e

        return infer_results



