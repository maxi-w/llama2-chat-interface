import fire
import gradio as gr
from llama import Llama
from typing import Optional


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [[]]

    def llama_response(message, history):
        dialogs[0].append({"role": "user", "content": message})

        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        dialogs[0].append(results[-1]["generation"])
        return results[-1]['generation']['content']

    demo = gr.ChatInterface(
        llama_response, 
        title="Llama 2 7B-chat",
        retry_btn=None,
        undo_btn=None,
        clear_btn=None,
    )

    demo.launch()

if __name__ == "__main__":
    fire.Fire(main)