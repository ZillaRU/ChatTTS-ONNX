import ChatTTS
import torch
import torchaudio
import numpy as np

chat = ChatTTS.Chat()
chat.load(local_path='./model_files')
wavs = chat.infer(["如果上天能再给我一次机会的话、我会对这个女孩说“我爱你”。如果非要在这份爱加上一个期限。我希望是,一万年!"], 
                  skip_refine_text=True, use_decoder=True)

torchaudio.save("test1.wav", wavs, sample_rate=24000)

# wavs = chat.infer(["如果上天能再给我一次机会的话、我会对这个女孩说“我爱你”。如果非要在这份爱加上一个期限。我希望是,一万年!"],
#                   params_refine_text = ChatTTS.Chat.RefineTextParams(
#                       prompt='[oral_2][laugh_0][break_6]',
#                       ),
#                   skip_refine_text=False, use_decoder=True, 
#                   params_infer_code = ChatTTS.Chat.InferCodeParams(
#                       prompt="[speed_5]",
#                       temperature=0.3,
#                       spk_emb=chat.sample_random_speaker_num()))
# torchaudio.save("test2.wav", wavs, sample_rate=24000)

wavs = chat.infer(["如果上天能再给我一次机会的话、我会对这个女孩说“我爱你”。如果非要在这份爱加上一个期限。我希望是,一万年!"], 
                  skip_refine_text=True, use_decoder=False)
torchaudio.save("test3.wav", wavs, sample_rate=24000)