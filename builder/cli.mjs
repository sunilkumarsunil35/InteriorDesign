#!/usr/bin/env node
// plan.json -> scene.json, for apps that want ready-made geometry in metres.
//   node builder/cli.mjs examples/2295-west/plan.json > scene.json
import { readFileSync } from 'node:fs';
import { planToScene } from './plan2scene.js';

const [, , input] = process.argv;
if (!input) {
  console.error('usage: node builder/cli.mjs plan.json > scene.json');
  process.exit(2);
}
const scene = planToScene(JSON.parse(readFileSync(input, 'utf8')));
for (const w of scene.warnings) console.error('warn:', w);
console.error(`rooms=${scene.rooms.length} walls=${scene.walls.length} openings=${scene.openings.length} furniture=${scene.furniture.length}`);
process.stdout.write(JSON.stringify(scene, null, 1));
