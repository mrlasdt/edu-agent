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
  await expect(page.getByRole("button", { name: "[1]" })).toBeVisible();

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
