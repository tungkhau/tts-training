import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


output_path = ''
dataset_path = "child_inclusion_male"

pretrained_path = 'vits_vctk-November-18-2024_08+49AM-0000000'

dataset_config = BaseDatasetConfig(
     meta_file_train="metadata.csv", language="it-it", path=dataset_path
)

# custom formatter implementation

def commonvoice_formatter(root_path, manifest_file, **kwargs):
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = dataset_path
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path,"wavs",cols[0]+ ".wav" )
            text = cols[1]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items


audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=23.0,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
    resample=True,
)

vitsArgs = VitsArgs(
    use_speaker_embedding=True,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="italian-tts",
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=False,
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    max_text_len=325,  # change this if you have a larger VRAM than 16GB
    output_path=output_path,
    datasets=[dataset_config],
)


if __name__ == '__main__':

    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_config, eval_split=True, formatter=commonvoice_formatter)

    # init speaker manager for multi-speaker training
    # it maps speaker-id to speaker-name in the model and data-loader
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.model_args.num_speakers = speaker_manager.num_speakers

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager)

    # init the trainer and 🚀
    if pretrained_path:
        trainer = Trainer(
            TrainerArgs(
                continue_path=pretrained_path,
            ),
            config,
            output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            gpu=0
        )
    else:
        trainer = Trainer(
            TrainerArgs(),
            config,
            output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            gpu=0
        )
    trainer.fit()