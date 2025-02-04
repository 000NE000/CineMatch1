## Database (DB) 
### def 
  - A structured system to store, manage, and retrieve data efficiently. 
  - It is a software that runs on a server (local, physical, or cloud) and allows applications to store and access data.
  - DBMS (Database Management System)
### 1) A database does not exist independently—it runs on some kind of server
	- Local Machine (Development)
      - You install PostgreSQL/MongoDB on your laptop.
      - Runs as a local process.
      - Good for development/testing.
	- Physical Server (On-Premise)
    	- A dedicated machine hosts the database.
    	- Usually managed by a team.
    	- Needs manual scaling.
	- Cloud Database (Managed)
    	- AWS RDS (PostgreSQL, MySQL), MongoDB Atlas, Google Cloud SQL.
    	- Fully managed, automatic scaling & backups.
    	- Best for production use.
### 2) How Data Moves in a Database (End-to-End)
  - 1st. Application (Client Side) Makes a Request
	- Ex: A user requests movie data in a web app.
  - 2nd. Server Processes the Request
    - Ex: A backend API receives the request.
  - 3rd. Query is Sent to the Database
    - The API converts the request into an SQL query (for PostgreSQL) or a NoSQL query (for MongoDB).
  - 4th. Database Processes the Query
	- The DBMS retrieves or stores data.
  - 5th. Data is Returned
	- The backend processes the result and sends it back to the frontend.
### SQL
- def 
  - SQL (Structured Query Language) is a language used to interact with a relational database—you use it to insert, retrieve, update, and delete data from the database.