// Host-only failure propagation controls. Fake model processes never touch a GPU.
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync, mkdirSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { spawnSync } from 'node:child_process';

const script = resolve(process.argv[2] ?? 'tests/e2e-gb10.sh');
const dir = mkdtempSync(join(tmpdir(), 'omega-e2e-gate-'));
try {
  const bin = join(dir, 'bin');
  mkdirSync(bin);
  const e2e = join(dir, 'fake-e2e');
  const cmp = join(dir, 'fake-cmp');
  writeFileSync(e2e, `#!/usr/bin/env node
const fs = require('node:fs');
fs.writeFileSync(process.argv[4], 'fixture');
console.log(JSON.stringify({step:0,token:1}));
console.log(JSON.stringify({n_vocab:2}));
if(process.env.OMEGA_GGML_DEVICES) {
  console.error('device 1: OmegaGB10');
  if(process.env.MOCK_PLACEMENT !== 'none') {
    for(const op of ['MUL_MAT','RMS_NORM','SWIGLU']) {
      if(process.env.MOCK_PLACEMENT === op) continue;
      console.error('omega-gb10-profile op='+op+' calls=1 ms=0');
    }
  }
  if(process.env.E2E_MAPS_OUT && process.env.MOCK_MAPS !== 'missing')
    fs.writeFileSync(process.env.E2E_MAPS_OUT, process.env.MOCK_MAPS === 'cuda' ? '/usr/lib/libcuda.so.1\\n' : '/usr/lib/libc.so.6\\n');
}
`, { mode: 0o755 });
  writeFileSync(cmp, `#!/usr/bin/env node
console.log(JSON.stringify({steps:1,top1_agree:1}));
process.exit(Number(process.env.MOCK_CMP_EXIT ?? 0));
`, { mode: 0o755 });
  writeFileSync(join(bin, 'cc'), `#!/bin/sh
set -eu
out=
source=
while test "$#" -gt 0; do
  case "$1" in
    -o) shift; out=$1 ;;
    tests/e2e.c) source=$MOCK_E2E ;;
    tests/cmp.c) source=$MOCK_CMP ;;
  esac
  shift
done
test -n "$out" && test -n "$source"
cp "$source" "$out"
chmod +x "$out"
`, { mode: 0o755 });
  let count = 0;
  function check(label, extra, pass) {
    const out = join(dir, 'case-'+count);
    const r = spawnSync('sh', [script, dir, 'fixture.gguf', 'fixture.so', '1', out], {
      encoding: 'utf8', env: {...process.env, PATH:bin+':'+process.env.PATH, MOCK_E2E:e2e, MOCK_CMP:cmp, ...extra},
    });
    assert.equal(r.error, undefined, label);
    assert.equal(r.signal, null, label);
    assert.equal(r.status === 0, pass, `${label}: exit=${r.status} ${r.stderr} ${r.stdout}`);
    if(pass) assert.ok(existsSync(join(out,'cmp.json')));
    console.log('PASS '+label);
    count++;
  }
  check('complete placed comparison', {}, true);
  check('comparator failure reaches caller', {MOCK_CMP_EXIT:'1'}, false);
  check('registered device without execution', {MOCK_PLACEMENT:'none'}, false);
  for(const op of ['MUL_MAT','RMS_NORM','SWIGLU'])
    check('missing executed '+op, {MOCK_PLACEMENT:op}, false);
  check('missing inference maps', {MOCK_MAPS:'missing'}, false);
  check('forbidden inference library', {MOCK_MAPS:'cuda'}, false);
  console.log(count+' full-model driver controls PASS');
} finally {
  rmSync(dir, { recursive: true, force: true });
}
