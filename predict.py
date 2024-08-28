from cog import BasePredictor, Input, Path
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["COQUI_TOS_AGREED"] = "1"
        config = XttsConfig()
        config.load_json("./config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, "./", eval=True)
        self.model.cuda()

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize",
            default="Hi there, I'm your new voice clone. Try your best to upload quality audio"
        ),
        speaker: Path = Input(description="Original speaker audio (wav, mp3, m4a, ogg, or flv). Duration should be at least 6 seconds."),
        language: str = Input(
            description="Output language for the synthesized speech",
            choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "hi"],
            default="en"
        ),
        cleanup_voice: bool = Input(
            description="Whether to apply denoising to the speaker audio (microphone recordings)",
            default=False
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        speaker_wav = "/tmp/speaker.wav"
        output_wav = "/tmp/output.wav"

        filter = "highpass=75,lowpass=8000,"
        trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02"
        
        # Convert and clean up the speaker audio if necessary
        if cleanup_voice:
            os.system(f"ffmpeg -i {speaker} -af {filter}{trim_silence} -y {speaker_wav}")
        else:
            os.system(f"ffmpeg -i {speaker} -y {speaker_wav}")

        # Load the cleaned speaker audio
        speaker_audio, sr = torchaudio.load(speaker_wav)
        
        # Synthesize speech
        synthesis_output = self.model.synthesize(
            text=text,
            config=self.model.config,
            speaker_wav=[speaker_wav],
            language=language
        )

        # Save the generated speech to a file
        torchaudio.save(output_wav, torch.tensor(synthesis_output["wav"]).unsqueeze(0), synthesis_output["config"].output_sample_rate)

        return Path(output_wav)
