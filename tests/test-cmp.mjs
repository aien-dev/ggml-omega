// Verification-only controls. No model, GPU, Python or production runtime.
// Byte fixtures sourced with WolframLanguageEvaluator, 2026-10-29:
// Reverse[IntegerDigits[h,256,4]] for h=16^^3F800000,16^^40000000,
// 16^^7FC00000,16^^7F800000 -> [0,0,128,63],[0,0,0,64],
// [0,0,192,127],[0,0,128,127]. First[Ordering[{1.,2.},-1]]-1 -> 1.
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const binary = resolve(process.argv[2] ?? 'build/cmp');
const dir = mkdtempSync(join(tmpdir(), 'omega-cmp-'));
let count = 0;
try {
  const one = Buffer.from([0, 0, 128, 63]), two = Buffer.from([0, 0, 0, 64]);
  const nan = Buffer.from([0, 0, 192, 127]), inf = Buffer.from([0, 0, 128, 127]);
  const row = Buffer.concat([one, two]);
  const files = {};
  for (const [name, bytes] of Object.entries({
    same: row, shifted: Buffer.concat([two, one]), empty: Buffer.alloc(0),
    short: row.subarray(0, 7), trailing: Buffer.concat([row, Buffer.from([0])]),
    twice: Buffer.concat([row, row]), nan: Buffer.concat([nan, two]), inf: Buffer.concat([one, inf]),
  })) {
    files[name] = join(dir, name);
    writeFileSync(files[name], bytes);
  }
  function check(label, a, b, pass, nv = '2', expected) {
    const args = [nv, a, b];
    if (expected !== undefined) args.push(expected);
    const r = spawnSync(binary, args, { encoding: 'utf8' });
    assert.equal(r.error, undefined, label);
    assert.equal(r.signal, null, `${label}: process died`);
    assert.equal(r.status === 0, pass, `${label}: exit ${r.status} ${r.stderr} ${r.stdout}`);
    if (pass) {
      const report = JSON.parse(r.stdout);
      assert.ok(report.steps > 0);
      assert.equal(report.top1_agree, report.steps);
      assert.equal(report.first_divergence, -1);
    }
    console.log(`PASS ${label}`);
    count++;
  }
  check('equal nonempty rows', files.same, files.same, true);
  check('explicit expected rows', files.twice, files.twice, true, '2', '2');
  check('changed greedy token', files.same, files.shifted, false);
  check('empty pair', files.empty, files.empty, false);
  check('short float row', files.short, files.short, false);
  check('trailing partial float', files.trailing, files.trailing, false);
  check('unequal complete rows', files.same, files.twice, false);
  check('equally truncated rows', files.same, files.same, false, '2', '2');
  check('extra rows', files.twice, files.twice, false, '2', '1');
  check('NaN', files.nan, files.nan, false);
  check('Infinity', files.inf, files.inf, false);
  check('missing file', join(dir, 'missing'), files.same, false);
  check('directory read error', dir, dir, false);
  for (const nv of ['0', '-1', '2junk', '', '999999999999999999999999999999'])
    check(`invalid vocabulary ${JSON.stringify(nv)}`, files.same, files.same, false, nv);
  for (const n of ['0', '-1', '1junk'])
    check(`invalid expected steps ${JSON.stringify(n)}`, files.same, files.same, false, '2', n);
  console.log(`${count} comparator controls PASS`);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
