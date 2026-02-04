#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio

# Reuse functions from dub3.py
from dub3 import translate_with_llama, generate_french_audio_from_subtitles


def main():
    parser = argparse.ArgumentParser(description="Translate subtitles to French and generate a single aligned audio file")
    parser.add_argument('-s', '--subtitle', required=True, help='Input subtitle file (SRT/VTT)')
    parser.add_argument('-o', '--output', help='Output audio path (mp3). Default: <subtitle>_audio.mp3')
    parser.add_argument('--subtitle-out', help='Path to save the translated subtitle file (optional). Default: <subtitle>_fr.<ext>')
    parser.add_argument('--skip-translate', action='store_true', help='If set, assume subtitle is already translated to French and skip translation')
    args = parser.parse_args()

    sub_in = args.subtitle
    if not os.path.exists(sub_in):
        print(f"Fichier introuvable: {sub_in}")
        sys.exit(2)

    base, ext = os.path.splitext(sub_in)
    out_audio = args.output or (base + "_audio.mp3")
    out_sub = args.subtitle_out or (base + "_fr" + (ext or ".srt"))

    if not args.skip_translate:
        print(f"Traduction du fichier de sous-titres '{sub_in}' vers '{out_sub}'...")
        text = None
        with open(sub_in, 'r', encoding='utf-8') as f:
            text = f.read()
        # translate_with_llama writes the output file and returns the translated content
        translate_with_llama(text, out_sub)
    else:
        out_sub = sub_in

    print(f"Génération de l'audio aligné (cela peut prendre du temps)...\nSortie: {out_audio}")
    # generate_french_audio_from_subtitles is async
    try:
        asyncio.run(generate_french_audio_from_subtitles(out_sub, out_audio))
    except Exception as e:
        print(f"Erreur pendant la génération audio: {e}")
        sys.exit(1)

    print("Terminé : audio généré.")


if __name__ == '__main__':
    main()
