import { expect, test } from "@playwright/test";

// The gateway is mocked: no agent service or API key is involved. The mocked
// response is fulfilled as a single body; the streaming reader still runs.
test("picks a subject, sends a message and renders the streamed answer", async ({
  page,
}) => {
  const requests: Record<string, unknown>[] = [];
  await page.route("**/chat", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({
      status: 200,
      contentType: "text/plain; charset=utf-8",
      body: "Try isolating x first [1].",
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Quant" }).click();
  await expect(page.getByRole("heading", { name: /Quant/ })).toBeVisible();

  await page.getByLabel("Your message").fill("Solve 2x + 3 = 11");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("Solve 2x + 3 = 11")).toBeVisible();
  await expect(page.getByText(/Try isolating x first/)).toBeVisible();
  // No sources panel yet, so the marker is plain text rather than a button that does nothing.
  await expect(page.getByText("[1]", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "[1]" })).toHaveCount(0);

  expect(requests[0]).toMatchObject({
    test_type: "gre",
    section: "quant",
    mode: "tutor",
    message: "Solve 2x + 3 = 11",
  });

  await page.getByRole("button", { name: "Show me the answer" }).click();
  await expect(
    page.getByRole("button", { name: "Back to hints" })
  ).toBeVisible();
});

test("failed and empty turns are not replayed as history", async ({ page }) => {
  const requests: Record<string, unknown>[] = [];
  const bodies = [
    { status: 500, body: "boom" },
    { status: 200, body: "" },
    { status: 200, body: "Fine." },
  ];
  await page.route("**/chat", async (route) => {
    requests.push(route.request().postDataJSON());
    const next = bodies[requests.length - 1];
    await route.fulfill({
      status: next.status,
      contentType: "text/plain; charset=utf-8",
      body: next.body,
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Quant" }).click();
  const input = page.getByLabel("Your message");
  const send = page.getByRole("button", { name: "Send" });

  await input.fill("first");
  await send.click();
  await expect(page.getByText("Chat request failed (500)")).toBeVisible();

  await input.fill("second");
  await send.click();
  await expect(page.getByText("No response received")).toBeVisible();

  await input.fill("third");
  await send.click();
  await expect(page.getByText("Fine.")).toBeVisible();

  expect(requests[2].history).toEqual([
    { role: "candidate", content: "first" },
    { role: "candidate", content: "second" },
  ]);
});

test("solve mode shows the math verifier's verdict and typeset math, not the raw trailer", async ({
  page,
}) => {
  await page.route("**/chat", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/plain; charset=utf-8",
      body: 'Factor x^2 - 5x + 6 = 0 => (x - 2)(x - 3).\n\n**Answer: x = 2, x = 3**\x1e{"math_verified":true}',
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Quant" }).click();
  await page.getByLabel("Your message").fill("Solve x^2 - 5x + 6 = 0");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("Answer: x = 2, x = 3")).toBeVisible();
  await expect(page.getByRole("note")).toContainText("checked by the math verifier");
  await expect(page.locator(".message-bubble--agent sup")).toHaveText("2");
  await expect(page.locator(".message-bubble--agent .message-content")).not.toContainText("math_verified");
});
