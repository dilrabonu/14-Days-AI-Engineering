import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import which
import os
import librosa
import soundfile as sf
from typing import List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class VADProcessor:
    """
    Complete Voice Activity Detection system using Silero VAD
    Perfect for logistics AI chatbot applications
    """
    
    def __init__(self, model_name: str = 'silero_vad'):
        """
        Initialize VAD processor
        
        Args:
            model_name: Name of the VAD model to use
        """
        self.model = None
        self.utils = None
        self.load_model(model_name)
        
    def load_model(self, model_name: str):
        """Load Silero VAD model"""
        try:
            # Load Silero VAD model
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model=model_name,
                force_reload=False,
                onnx=False
            )
            
            # Extract utility functions
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = self.utils
             
            print(" Silero VAD model loaded successfully!")
            
        except Exception as e:
            print(f" Error loading VAD model: {e}")
            raise
    
    def validate_and_convert_audio(self, input_path: str, output_path: str = None) -> str:
        """
        Validate and convert audio file to proper format
        
        Args:
            input_path: Path to input audio file
            output_path: Path for converted file (optional)
            
        Returns:
            Path to valid audio file
        """
        try:
            # First, try to detect file format and validate
            file_info = self._get_file_info(input_path)
            print(f" Original file info: {file_info}")
            
            # Try different loading methods
            audio_data = None
            sample_rate = None
            
            # Method 1: Try librosa (most robust)
            try:
                audio_data, sample_rate = librosa.load(input_path, sr=None)
                print(f" Loaded with librosa: SR={sample_rate}, Duration={len(audio_data)/sample_rate:.2f}s")
            except Exception as e:
                print(f" Librosa failed: {e}")
                
                # Method 2: Try torchaudio
                try:
                    audio_data, sample_rate = torchaudio.load(input_path)
                    audio_data = audio_data.numpy().flatten()
                    print(f" Loaded with torchaudio: SR={sample_rate}, Duration={len(audio_data)/sample_rate:.2f}s")
                except Exception as e:
                    print(f" Torchaudio failed: {e}")
                    
                    # Method 3: Try soundfile
                    try:
                        audio_data, sample_rate = sf.read(input_path)
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data.mean(axis=1)  # Convert to mono
                        print(f" Loaded with soundfile: SR={sample_rate}, Duration={len(audio_data)/sample_rate:.2f}s")
                    except Exception as e:
                        print(f" All loading methods failed: {e}")
                        raise ValueError(f"Cannot load audio file: {input_path}")
            
            # Convert to 16kHz mono if needed (Silero VAD requirement)
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
                print(f" Resampled to 16kHz")
            
            # Save converted file if output path provided
            if output_path:
                sf.write(output_path, audio_data, sample_rate)
                print(f" Saved converted audio to: {output_path}")
                return output_path
            else:
                # Save temporary file
                temp_path = input_path.replace('.wav', '_converted.wav')
                sf.write(temp_path, audio_data, sample_rate)
                return temp_path
                
        except Exception as e:
            print(f" Error in audio validation/conversion: {e}")
            raise
    
    def _get_file_info(self, file_path: str) -> dict:
        """Get basic file information"""
        try:
            file_size = os.path.getsize(file_path)
            
            # Try to read first few bytes to check format
            with open(file_path, 'rb') as f:
                header = f.read(12)
                
            return {
                'size_bytes': file_size,
                'size_mb': file_size / (1024 * 1024),
                'header_hex': header.hex() if header else 'None',
                'is_wav': header.startswith(b'RIFF') and b'WAVE' in header if header else False
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_speech_segments(self, audio_path: str, 
                             threshold: float = 0.5,
                             min_speech_duration_ms: int = 250,
                             min_silence_duration_ms: int = 100) -> List[dict]:
        """
        Detect speech segments in audio file
        
        Args:
            audio_path: Path to audio file
            threshold: VAD confidence threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence duration between segments
            
        Returns:
            List of speech segments with timestamps
        """
        try:
            # Validate and load audio
            converted_path = self.validate_and_convert_audio(audio_path)
            wav = self.read_audio(converted_path, sampling_rate=16000)
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                wav, 
                self.model,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                return_seconds=True
            )
            
            print(f" Detected {len(speech_timestamps)} speech segments")
            
            # Add additional information
            segments = []
            total_duration = len(wav) / 16000  # Total duration in seconds
            
            for i, segment in enumerate(speech_timestamps):
                segments.append({
                    'segment_id': i + 1,
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'duration': segment['end'] - segment['start'],
                    'confidence': threshold  # You can enhance this with actual confidence scores
                })
            
            # Calculate statistics
            total_speech_duration = sum(seg['duration'] for seg in segments)
            speech_ratio = total_speech_duration / total_duration if total_duration > 0 else 0
            
            print(f" Speech Statistics:")
            print(f"   Total duration: {total_duration:.2f}s")
            print(f"   Speech duration: {total_speech_duration:.2f}s")
            print(f"   Speech ratio: {speech_ratio:.1%}")
            
            # Clean up temporary file
            if converted_path != audio_path and os.path.exists(converted_path):
                os.remove(converted_path)
            
            return segments
            
        except Exception as e:
            print(f" Error in speech detection: {e}")
            raise
    
    def extract_speech_segments(self, audio_path: str, 
                              output_dir: str = "speech_segments",
                              **vad_params) -> List[str]:
        """
        Extract and save individual speech segments
        
        Args:
            audio_path: Path to input audio
            output_dir: Directory to save segments
            **vad_params: Parameters for VAD detection
            
        Returns:
            List of paths to extracted segments
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Detect speech segments
            segments = self.detect_speech_segments(audio_path, **vad_params)
            
            # Load audio for segmentation
            converted_path = self.validate_and_convert_audio(audio_path)
            audio_data, sr = librosa.load(converted_path, sr=16000)
            
            extracted_files = []
            
            for segment in segments:
                # Calculate sample indices
                start_sample = int(segment['start_time'] * sr)
                end_sample = int(segment['end_time'] * sr)
                
                # Extract segment
                segment_audio = audio_data[start_sample:end_sample]
                
                # Save segment
                filename = f"segment_{segment['segment_id']:03d}_{segment['start_time']:.2f}s-{segment['end_time']:.2f}s.wav"
                output_path = os.path.join(output_dir, filename)
                
                sf.write(output_path, segment_audio, sr)
                extracted_files.append(output_path)
                
                print(f" Saved: {filename} (duration: {segment['duration']:.2f}s)")
            
            # Clean up temporary file
            if converted_path != audio_path and os.path.exists(converted_path):
                os.remove(converted_path)
            
            return extracted_files
            
        except Exception as e:
            print(f" Error extracting segments: {e}")
            raise
    
    def create_clean_audio(self, audio_path: str, 
                          output_path: str,
                          padding_ms: int = 100,
                          **vad_params) -> str:
        """
        Create clean audio with only speech parts (remove silence)
        
        Args:
            audio_path: Input audio path
            output_path: Output clean audio path
            padding_ms: Padding around speech segments in milliseconds
            **vad_params: VAD parameters
            
        Returns:
            Path to clean audio file
        """
        try:
            # Detect speech segments
            segments = self.detect_speech_segments(audio_path, **vad_params)
            
            if not segments:
                print(" No speech segments detected!")
                return None
            
            # Load audio
            converted_path = self.validate_and_convert_audio(audio_path)
            audio_data, sr = librosa.load(converted_path, sr=16000)
            
            # Extract and concatenate speech segments with padding
            clean_audio = []
            padding_samples = int(padding_ms * sr / 1000)
            
            for segment in segments:
                start_sample = max(0, int(segment['start_time'] * sr) - padding_samples)
                end_sample = min(len(audio_data), int(segment['end_time'] * sr) + padding_samples)
                
                segment_audio = audio_data[start_sample:end_sample]
                clean_audio.extend(segment_audio)
            
            # Save clean audio
            clean_audio = np.array(clean_audio)
            sf.write(output_path, clean_audio, sr)
            
            # Calculate compression ratio
            original_duration = len(audio_data) / sr
            clean_duration = len(clean_audio) / sr
            compression_ratio = clean_duration / original_duration
            
            print(f" Clean audio created:")
            print(f"   Original: {original_duration:.2f}s")
            print(f"   Clean: {clean_duration:.2f}s")
            print(f"   Compression: {compression_ratio:.1%}")
            print(f"   Saved to: {output_path}")
            
            # Clean up temporary file
            if converted_path != audio_path and os.path.exists(converted_path):
                os.remove(converted_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error creating clean audio: {e}")
            raise
    
    def visualize_vad_results(self, audio_path: str, 
                             segments: List[dict] = None,
                             save_plot: bool = True,
                             **vad_params):
        """
        Visualize VAD results with waveform and detected segments
        """
        try:
            # Load audio for visualization
            converted_path = self.validate_and_convert_audio(audio_path)
            audio_data, sr = librosa.load(converted_path, sr=16000)
            
            # Get segments if not provided
            if segments is None:
                segments = self.detect_speech_segments(audio_path, **vad_params)
            
            # Create time axis
            time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
            
            # Create plot
            plt.figure(figsize=(15, 8))
            
            # Plot waveform
            plt.subplot(2, 1, 1)
            plt.plot(time_axis, audio_data, color='blue', alpha=0.7, linewidth=0.5)
            plt.title('Audio Waveform with VAD Results', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            # Highlight speech segments
            for segment in segments:
                plt.axvspan(segment['start_time'], segment['end_time'], 
                           color='red', alpha=0.3, label='Speech' if segment == segments[0] else "")
            
            if segments:
                plt.legend()
            
            # Plot VAD decisions
            plt.subplot(2, 1, 2)
            vad_timeline = np.zeros(len(time_axis))
            
            for segment in segments:
                start_idx = int(segment['start_time'] * sr)
                end_idx = int(segment['end_time'] * sr)
                start_idx = max(0, min(start_idx, len(vad_timeline) - 1))
                end_idx = max(0, min(end_idx, len(vad_timeline)))
                vad_timeline[start_idx:end_idx] = 1
            
            plt.plot(time_axis, vad_timeline, color='red', linewidth=2)
            plt.title('VAD Decisions (1=Speech, 0=Silence)', fontsize=12)
            plt.xlabel('Time (seconds)')
            plt.ylabel('VAD Decision')
            plt.ylim(-0.1, 1.1)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = audio_path.replace('.wav', '_vad_analysis.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f" Plot saved to: {plot_path}")
            
            plt.show()
            
            # Clean up temporary file
            if converted_path != audio_path and os.path.exists(converted_path):
                os.remove(converted_path)
            
        except Exception as e:
            print(f" Error in visualization: {e}")
            raise

# Example usage and testing
def main():
    """
    Main function demonstrating VAD usage for logistics AI chatbot
    """
    print(" Starting Voice Activity Detection Analysis")
    print("=" * 50)
    
    # Initialize VAD processor
    try:
        vad_processor = VADProcessor()
    except Exception as e:
        print(f" Failed to initialize VAD processor: {e}")
        return
    
    # Define your audio file path
    # Replace this with your actual audio file path
    audio_file = "day3.wav"  # Change this to your file
    
    if not os.path.exists(audio_file):
        print(f" Audio file not found: {audio_file}")
        print(" Creating a sample audio file for testing...")
        
        # Create a simple test audio file
        create_test_audio(audio_file)
    
    try:
        # 1. Detect speech segments
        print("\n1️ Detecting speech segments...")
        segments = vad_processor.detect_speech_segments(
            audio_file,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )
        
        # 2. Extract individual segments
        print("\n2️ Extracting speech segments...")
        extracted_files = vad_processor.extract_speech_segments(
            audio_file,
            output_dir="extracted_speech"
        )
        
        # 3. Create clean audio (silence removed)
        print("\n3️ Creating clean audio...")
        clean_audio_path = vad_processor.create_clean_audio(
            audio_file,
            "clean_speech.wav",
            padding_ms=100
        )
        
        # 4. Visualize results
        print("\n4️ Creating visualization...")
        vad_processor.visualize_vad_results(audio_file, segments)
        
        print("\n VAD Analysis Complete!")
        print(f"   - Detected {len(segments)} speech segments")
        print(f"   - Extracted {len(extracted_files)} segment files")
        print(f"   - Created clean audio: {clean_audio_path}")
        
    except Exception as e:
        print(f" Error during processing: {e}")
        import traceback
        traceback.print_exc()

def create_test_audio(output_path: str):
    """Create a test audio file with speech and silence"""
    try:
        import numpy as np
        
        # Generate test audio: 3 seconds of noise, 2 seconds silence, 3 seconds noise
        sr = 16000
        
        # Speech-like signal (filtered noise)
        speech1 = np.random.normal(0, 0.1, sr * 3)  # 3 seconds
        speech1 = np.convolve(speech1, np.ones(100)/100, mode='same')  # Smooth
        
        # Silence
        silence = np.zeros(sr * 2)  # 2 seconds
        
        # Another speech segment
        speech2 = np.random.normal(0, 0.08, sr * 2)  # 2 seconds
        speech2 = np.convolve(speech2, np.ones(80)/80, mode='same')
        
        # Combine
        test_audio = np.concatenate([speech1, silence, speech2])
        
        # Save
        sf.write(output_path, test_audio, sr)
        print(f" Created test audio file: {output_path}")
        
    except Exception as e:
        print(f" Error creating test audio: {e}")

if __name__ == "__main__":
    main()