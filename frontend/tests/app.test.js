const test = require("node:test");
const assert = require("node:assert/strict");

const {
  resolveApiUrl,
  formatPercent,
  probabilityRowHtml,
  probabilitiesTableHtml,
} = require("../app.js");

test("resolveApiUrl targets the local backend when served from localhost", () => {
  assert.equal(resolveApiUrl("localhost", "http:"), "http://localhost:8000/predict");
  assert.equal(resolveApiUrl("127.0.0.1", "http:"), "http://localhost:8000/predict");
});

test("resolveApiUrl targets the same host's backend port otherwise", () => {
  assert.equal(resolveApiUrl("192.168.1.10", "http:"), "http://192.168.1.10:8000/predict");
  assert.equal(resolveApiUrl("my-app.example.com", "https:"), "https://my-app.example.com:8000/predict");
});

test("formatPercent renders one decimal place as a percentage", () => {
  assert.equal(formatPercent(0.8777), "87.8%");
  assert.equal(formatPercent(0), "0.0%");
  assert.equal(formatPercent(1), "100.0%");
});

test("probabilityRowHtml embeds the digit and formatted probability", () => {
  const html = probabilityRowHtml(7, 0.5);
  assert.match(html, /<td class="p-digit">7<\/td>/);
  assert.match(html, /50\.0%/);
});

test("probabilitiesTableHtml renders exactly one row per probability", () => {
  const html = probabilitiesTableHtml([0.1, 0.9]);
  const rowCount = (html.match(/<tr>/g) || []).length;
  assert.equal(rowCount, 2);
});

test("probabilitiesTableHtml handles all ten MNIST classes", () => {
  const probabilities = new Array(10).fill(0.1);
  const html = probabilitiesTableHtml(probabilities);
  for (let digit = 0; digit < 10; digit += 1) {
    assert.match(html, new RegExp(`<td class="p-digit">${digit}</td>`));
  }
});
