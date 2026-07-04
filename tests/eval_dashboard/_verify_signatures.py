import dataclasses
import inspect

import hexfield.config as c
import hexfield.eval_arena as a
import hexfield.multistage_eval as m

print("select_opponents:", inspect.signature(m.select_opponents))
print("_stage_d_pool:", inspect.signature(m._stage_d_pool))
print("_build_checkpoint_edge_from_match:", inspect.signature(m._build_checkpoint_edge_from_match))
print("_choose_anchor:", inspect.signature(m._choose_anchor))
print("_write_eval_hxr:", inspect.signature(a._write_eval_hxr))
print("run_multistage_eval_concurrent present:", hasattr(m, "run_multistage_eval_concurrent"))
print("Roster fields:", [f.name for f in dataclasses.fields(m.Roster)])
print("radius8_opponents default:", c.MultiStageEvalOpponents().radius8_opponents)
print("SIGCHECK OK")
