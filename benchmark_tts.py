import importlib.util
import argparse
import asyncio
import time
import json
import os
import statistics
import io

spec = importlib.util.spec_from_file_location("dub4", r"c:\Users\ratah\OneDrive\Documents\traducteur\dub4.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

async def run_benchmark(mode, n_segments=100, concurrency=6, block_concurrency=4, timeout=30, max_attempts=2):
    print(f"Benchmark mode={mode} n_segments={n_segments} concurrency={concurrency} block_concurrency={block_concurrency}")
    # parse srt and pick first n_segments
    subs = mod.parse_srt(mod.SUBTITLE_PATH if hasattr(mod, 'SUBTITLE_PATH') else 'subtitle.srt')
    subs = subs[:n_segments]

    # translate all segments (we'll use Google to be consistent)
    print("Translating segments (concurrent)...")
    t0 = time.time()
    translations = await asyncio.gather(*[asyncio.create_task(mod.translate_text_async(s['text'], 'google')) for s in subs])
    t_trans = time.time() - t0
    print(f"Translation done in {t_trans:.2f}s")

    sem = asyncio.BoundedSemaphore(concurrency)

    async def process_one(idx, text):
        # prepare params
        target_duration = max(0.1, subs[idx]['end'] - subs[idx]['start'])
        voice = 'fr-FR-HenriNeural'
        # tts
        t0 = time.time()
        if mode == 'memory':
            res = await asyncio.wait_for(mod.generate_tts_synced(text, voice, target_duration, None, sem, timeout_per_attempt=timeout, max_attempts=max_attempts, return_bytes=True), timeout=timeout+10)
        else:
            # disk mode: write to a temp file
            import tempfile
            tmpf = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            tmpf.close()
            try:
                res = await asyncio.wait_for(mod.generate_tts_synced(text, voice, target_duration, tmpf.name, sem, timeout_per_attempt=timeout, max_attempts=max_attempts, return_bytes=False), timeout=timeout+10)
            finally:
                pass
        t_tts = time.time() - t0

        # split using pydub (works for both memory bytes and on-disk files)
        t0 = time.time()
        ok = False
        try:
            from pydub import AudioSegment
            if mode == 'memory' and res.get('audio_bytes'):
                audio = AudioSegment.from_file(io.BytesIO(res.get('audio_bytes')), format='mp3')
            else:
                audio = AudioSegment.from_file(tmpf.name, format='mp3')
            start_ms = 0
            seg_dur_ms = int(max(0.1, subs[idx]['end'] - subs[idx]['start']) * 1000)
            end_ms = min(start_ms + seg_dur_ms, len(audio))
            chunk = audio[start_ms:end_ms]
            seg_path = os.path.join(mod.TEMP_DIR, f"bench_seg_{idx:04d}.mp3")
            chunk.export(seg_path, format='mp3')
            ok = os.path.exists(seg_path)
            t_split = time.time() - t0
        except Exception as e:
            t_split = time.time() - t0
            ok = False
            print('Split error:', e)
        finally:
            # cleanup tmp file
            if mode != 'memory' and 'tmpf' in locals():
                try:
                    os.remove(tmpf.name)
                except Exception:
                    pass
        return {'idx': idx, 'tts_time': t_tts, 'split_time': t_split, 'ok': ok, 'duration': res.get('duration')}

    # run tasks with block_concurrency
    sem_block = asyncio.BoundedSemaphore(block_concurrency)
    async def worker(i, text):
        async with sem_block:
            return await process_one(i, text)

    tasks = [asyncio.create_task(worker(i, translations[i])) for i in range(len(subs))]
    t0 = time.time()
    results = await asyncio.gather(*tasks)
    total = time.time() - t0

    # compute stats
    tts_times = [r['tts_time'] for r in results if r['tts_time'] is not None]
    split_times = [r['split_time'] for r in results if r['split_time'] is not None]
    durations = [r['duration'] for r in results if r['duration']]

    summary = {
        'mode': mode,
        'n_segments': n_segments,
        'concurrency': concurrency,
        'block_concurrency': block_concurrency,
        'total_time': total,
        'translate_time': t_trans,
        'avg_tts': statistics.mean(tts_times) if tts_times else None,
        'p50_tts': statistics.median(tts_times) if tts_times else None,
        'avg_split': statistics.mean(split_times) if split_times else None,
        'p50_split': statistics.median(split_times) if split_times else None,
        'success_rate': sum(1 for r in results if r['ok']) / len(results)
    }

    return {'summary': summary, 'results': results}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segments', type=int, default=100)
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--block-concurrency', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--max-attempts', type=int, default=2)
    args = parser.parse_args()

    # ensure temp dir
    os.makedirs(mod.TEMP_DIR, exist_ok=True)

    async def main():
        res_mem = await run_benchmark('memory', n_segments=args.segments, concurrency=args.concurrency, block_concurrency=args.block_concurrency, timeout=args.timeout, max_attempts=args.max_attempts)
        print('Memory run summary:', res_mem['summary'])
        with open('benchmark_memory.json', 'w', encoding='utf-8') as fh:
            json.dump(res_mem, fh, ensure_ascii=False, indent=2)

        res_disk = await run_benchmark('disk', n_segments=args.segments, concurrency=args.concurrency, block_concurrency=args.block_concurrency, timeout=args.timeout, max_attempts=args.max_attempts)
        print('Disk run summary:', res_disk['summary'])
        with open('benchmark_disk.json', 'w', encoding='utf-8') as fh:
            json.dump(res_disk, fh, ensure_ascii=False, indent=2)

    asyncio.run(main())