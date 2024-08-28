# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio
import torch

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["COQUI_TOS_AGREED"] = "1"
        config = XttsConfig()
        config.load_json("./config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, "./", vocab_path="./vocab.json", speaker_file="./speakers_xtts.pth", eval=True)
        self.model.cuda()

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize",
            default="Hi there, I'm your new voice clone. Try your best to upload quality audio"
        ),
        speaker: Path = Input(description="Original speaker audio (wav, mp3, m4a, ogg, or flv). Duration should be at least 6 seconds."),
        language: str = Input(
            description="Output language for the synthesised speech",
            choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "hu", "ko", "hi"],
            default="en"
        ),
        temperature: float = Input(
            description="Temperature for the synthesised speech",
            default=0.75,
            ge=0.01,
            le=1.0
        ),
        length_penalty: float = Input(
            description="Length penalty for the synthesised speech",
            default=1.0,
            ge=-10.0,
            le=10.0
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty for the synthesised speech",
            default=10.0,
            ge=1,
            le=10
        ),
        top_k: int = Input(
            description="Top k for the synthesised speech",
            default=50,
            ge=1,
            le=100
        ),
        top_p: float = Input(
            description="Top p for the synthesised speech",
            default=0.85,
            ge=0.01,
            le=1.0
        ),
        do_sample: bool = Input(
            description="Whether to sample from the synthesised speech",
            default=True
        ),
        gpt_cond_len: int = Input(
            description="GPT conditioning length for the synthesised speech",
            default=30,
            ge=20,
            le=30
        ),
        gpt_cond_chunk_len: int = Input(
            description="GPT conditioning chunk length for the synthesised speech",
            default=6,
            ge=5,
            le=10
        ),
        max_ref_len: int = Input(
            description="Maximum reference length for the synthesised speech",
            default=30,
            ge=10,
            le=60
        ),
        sound_norm_: bool = Input(
            description="Whether to apply sound normalization to the synthesised speech",
            default=False
        ),
        cleanup_voice: bool = Input(
            description="Whether to apply denoising to the speaker audio (microphone recordings)",
            default=False
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        speaker_wav = "/tmp/speaker.wav"
        filter = "highpass=75,lowpass=8000,"
        trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02"
        # ffmpeg convert to wav and apply afftn denoise filter. y to overwrite and avoid caching
        if cleanup_voice:
            os.system(f"ffmpeg -i {speaker} -af {filter}{trim_silence} -y {speaker_wav}")
        else:
            os.system(f"ffmpeg -i {speaker} -y {speaker_wav}")

        # path = self.model.tts_to_file(
        #     text=text, 
        #     file_path = "/tmp/output.wav",
        #     speaker_wav = speaker_wav,
        #     language = language
        # )
        #

        out = self.model.full_inference(
            text,
            ref_audio_path=speaker_wav,
            language=language,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=gpt_cond_chunk_len,
            max_ref_len=max_ref_len,
            sound_norm_refs=sound_norm_
        )

        path = "/tmp/output.wav"
        torchaudio.save(
            path,
            torch.tensor(out["wav"]).unsqueeze(0),
            24000,
            format="wav",
        )

        return Path(path)
