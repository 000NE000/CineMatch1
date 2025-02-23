### Key Concepts:
1.	Event Loop:
•	The event loop is responsible for scheduling and managing asynchronous tasks.
•	It allows tasks to be paused (e.g., while waiting for an HTTP response) and resumed later.
•	This avoids blocking the program, allowing other tasks to proceed.
2.	async and await Keywords:
•	async is used to define an asynchronous function.
•	await is used to pause the function’s execution until the awaited asynchronous task completes.
•	Other tasks can be executed while awaiting, making the code non-blocking.