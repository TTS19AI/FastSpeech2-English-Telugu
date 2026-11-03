"""Gradio app; original assignment inference scripts remain unchanged."""
import atexit
import logging
import multiprocessing as mp
import os
from pathlib import Path
import threading

import gradio as gr
from deploy.bootstrap import prepare
from deploy.frontend import validate_text
from deploy.worker import run_worker

_lock = threading.Lock()
_workers = {}
_paths = None


def stop_workers():
    for process, connection in list(_workers.values()):
        connection.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
    _workers.clear()


atexit.register(stop_workers)


def generate(text, language):
    global _paths
    try:
        text = validate_text(text, language)
        with _lock:
            if _paths is None:
                _paths = prepare()
            if language not in _workers:
                context = mp.get_context('spawn')
                parent, child = context.Pipe()
                process = context.Process(target=run_worker, args=(child, language, _paths[language]), daemon=True)
                process.start()
                child.close()
                _workers[language] = (process, parent)
                if not parent.poll(300):
                    stop_workers()
                    raise RuntimeError('Model initialization timed out. Check hosting logs.')
                ready = parent.recv()
                if 'error' in ready:
                    stop_workers()
                    raise RuntimeError(ready['error'])
            process, connection = _workers[language]
            connection.send({'text': text})
            if not connection.poll(180):
                stop_workers()
                raise RuntimeError('Generation timed out. Please try a shorter sentence.')
            result = connection.recv()
            if 'error' in result:
                raise ValueError(result['error'])
            return result['path'], result['status']
    except Exception as exc:
        logging.exception('Speech generation failed')
        if not isinstance(exc, ValueError):
            stop_workers()
        raise gr.Error(str(exc)) from exc


def build_app():
    with gr.Blocks(title='English and Telugu Speech', delete_cache=(3600, 3600)) as demo:
        gr.Markdown('# English and Telugu Text-to-Speech\nWrite a short sentence, select its language, and generate speech.')
        language = gr.Dropdown(['English', 'Telugu'], value='English', label='Language')
        text = gr.Textbox(label='Text', lines=4, placeholder='Hello. Welcome to our speech synthesis demo.')
        gr.Markdown('Use up to 250 characters. Write numbers and times as words. For Telugu, use Telugu script. First use downloads and loads the models and can take several minutes.')
        button = gr.Button('Generate speech', variant='primary')
        audio = gr.Audio(label='Generated speech', type='filepath', format='wav')
        status = gr.Textbox(label='Status', interactive=False)
        gr.Examples([['Hello. Welcome to our speech synthesis demo.', 'English'],
                     ['తాడుపై అతని పట్టు చాలా బలంగా ఉంది.', 'Telugu']], [text, language])
        gr.Markdown('Research demo: English may sound partly noisy. Pronunciation of new words is not guaranteed. This app does not retrain the models.')
        button.click(generate, inputs=[text, language], outputs=[audio, status], concurrency_limit=1)
    return demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == '__main__':
    build_app().launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', '7860')),
                       allowed_paths=[str(Path('/tmp/fastspeech2_web_audio'))], show_error=True)
