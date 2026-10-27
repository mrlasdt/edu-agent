# MVP live check

A tutor-mode question asked in the browser at http://localhost:5173 against the stack from
`docker compose -f infra/docker-compose.yml up -d --build` (gpt-4o-mini, 28 seeded chunks in Qdrant):

![Tutor answer in the web app](chat-tutor.png)

The answer is grounded in the seeded notes ("According to your notes ...") and arrives as a
stream. Transcript of the same question through the web container's `/chat` proxy, with the
arrival time of each chunk, followed by a solve-mode turn:

```
[+ 3.0s] 'Great! '
[+ 3.2s] 'To start solving the equation x^2 - 5x + 6 = 0, you can use the factoring method. '
[+ 3.5s] 'According to your notes, look for two numbers that multiply to 6 (the constant term) and add up to -5 (the coefficient of x). \n\n'
[+ 4.6s] 'What are the two numbers that fit those criteria?'

--- solve mode ---
To solve the quadratic equation x^2 - 5x + 6 = 0, we will use the factoring method as outlined in the reference notes.

### Step-by-Step Solution:

1. **Identify the coefficients**: 
   In the equation x^2 - 5x + 6 = 0, we have:
   - a = 1 (coefficient of x^2)
   - b = -5 (coefficient of x)
   - c = 6 (constant term)

2. **Set up for factoring**: 
   We need to find two numbers that multiply to c (which is 6) and add to b (which is -5). 

3. **Find the numbers**: 
   The pairs of factors of 6 are:
   - 1 and 6
   - 2 and 3
   We need to check which pair adds up to -5. 

   The correct pair is:
   - -2 and -3 (since -2 * -3 = 6 and -2 + -3 = -5)

4. **Write the factored form**: 
   Using the numbers we found, we can factor the quadratic:
   (x - 2)(x - 3) = 0

5. **Set each factor to zero**: 
   To find the values of x, we set each factor equal to zero:
   - x - 2 = 0  →  x = 2
   - x - 3 = 0  →  x = 3

6. **List the solutions**: 
   The solutions to the equation x^2 - 5x + 6 = 0 are x = 2 and x = 3.

### Conclusion:
Thus, the final answer is:
**Answer: 2, 3**

--- corpus points ---
28 chunks
```
