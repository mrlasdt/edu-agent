import { expect, test } from "@playwright/test";

// The gateway is mocked: /chat answers with an X-Trace-Id and /turns/<id> returns the record.
const record = (over: Record<string, unknown> = {}) => ({
  trace_id: "t1",
  agent: "quant",
  mode: "solve",
  llm_calls: 3,
  input_tokens: 100,
  output_tokens: 30,
  estimated: false,
  cost_usd: 0.002,
  total_ms: 2100,
  calls: [],
  math_tools: true,
  tool_calls: [
    { name: "math_evaluate", ok: true, ms: 4, round: 1, arguments: '{"expression":"(1+15/100)*(1-20/100)"}', result: '{"value":"0.92"}' },
  ],
  tools_hidden: false,
  calc_check: { verdict: "agrees", answer: "-8%", tool_value: "0.92", tool: "math_evaluate" },
  ...over,
});

test("the Calculator switch changes the request and Show working lists the calculator", async ({ page }) => {
  const requests: Record<string, unknown>[] = [];
  await page.route("**/chat", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({
      status: 200,
      contentType: "text/plain; charset=utf-8",
      headers: { "X-Trace-Id": "t1" },
      body: "The net change is -8%.",
    });
  });
  await page.route("**/turns/t1", (route) => route.fulfill({ json: record() }));

  await page.goto("/");
  await page.getByRole("button", { name: "Quant" }).click();
  await page.getByRole("switch", { name: "Calculator: off" }).click();
  await expect(page.getByRole("switch", { name: "Calculator: on" })).toBeVisible();
  await page.getByLabel("Your message").fill("15% up then 20% down?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("The net change is -8%.")).toBeVisible();
  expect(requests[0]).toMatchObject({ math_tools: true });

  await page.getByText("Show working").click();
  await expect(page.getByText("math_evaluate (1+15/100)*(1-20/100) -> 0.92")).toBeVisible();
  await expect(page.getByText("Calculator agrees: answer -8% = 0.92")).toBeVisible();
});

test("tutor mode hides the calculator result until Reveal", async ({ page }) => {
  await page.route("**/chat", (route) =>
    route.fulfill({ status: 200, contentType: "text/plain", headers: { "X-Trace-Id": "t1" }, body: "Try factoring." })
  );
  await page.route("**/turns/t1*", (route) => {
    const revealed = route.request().url().includes("reveal=1");
    route.fulfill({
      json: record({
        mode: "tutor",
        tools_hidden: !revealed,
        calc_check: null,
        tool_calls: [
          revealed
            ? { name: "math_solve", ok: true, ms: 3, round: 1, arguments: '{"equation":"x^2-3x-4=0"}', result: '{"roots":["-1","4"]}' }
            : { name: "math_solve", ok: true, ms: 3, round: 1, arguments: null, result: null },
        ],
      }),
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Quant" }).click();
  await page.getByLabel("Your message").fill("x^2 - 3x - 4 = 0?");
  await page.getByRole("button", { name: "Send" }).click();
  await page.getByText("Show working").click();
  await expect(page.getByText(/Result hidden in tutor mode/)).toBeVisible();
  await expect(page.getByText(/roots/)).toHaveCount(0);
  await page.getByRole("button", { name: "Reveal" }).click();
  await expect(page.getByText("math_solve x^2-3x-4=0 -> roots -1, 4")).toBeVisible();
});
