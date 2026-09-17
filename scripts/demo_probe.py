"""Read one video header with a process-level timeout supplied by the caller."""
import json, sys
import av
av.logging.restore_default_callback()
with av.open(sys.argv[1]) as container:
    stream=container.streams.video[0]
    duration=float(stream.duration*stream.time_base) if stream.duration is not None else (container.duration/av.time_base if container.duration else None)
    print(json.dumps({'duration_s':duration,'fps':float(stream.average_rate or 0),'width':stream.width,'height':stream.height,'codec':stream.codec_context.name}))
