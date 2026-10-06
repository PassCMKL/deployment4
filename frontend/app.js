function resolveApiUrl(hostname, protocol) {
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://localhost:8000/predict";
  }
  return `${protocol}//${hostname}:8000/predict`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function probabilityRowHtml(digit, probability) {
  return `
        <tr>
          <td class="p-digit">${digit}</td>
          <td class="p-bar"><div class="bar-track"><div class="bar-fill" style="width:${formatPercent(probability)}"></div></div></td>
          <td class="p-val">${formatPercent(probability)}</td>
        </tr>
      `;
}

function probabilitiesTableHtml(probabilities) {
  return probabilities.map((p, i) => probabilityRowHtml(i, p)).join("");
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    resolveApiUrl,
    formatPercent,
    probabilityRowHtml,
    probabilitiesTableHtml,
  };
}
