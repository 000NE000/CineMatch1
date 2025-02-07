### Database (DB) 
- def 
  - A structured system to store, manage, and retrieve data efficiently. 
  - It is a software that runs on a server (local, physical, or cloud) and allows applications to store and access data.
  - DBMS (Database Management System)
- 1) A database does not exist independently—it runs on some kind of server
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
- 2) How Data Moves in a Database (End-to-End)
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

### DB 만들기
#### 스키마(Schema) 디자인
- 스키마는 DB의 구조와 제약 조건에 관한 전반적인 명세를 정의한 메타데이터의 집합
- 메타데이터(meta data)는 데이터에 대한 데이터로, 어떤 목적을 가지고 만들어졌는지 설명합니다.
개체의 특성을 나타내는 속성(Attribute)과, 속성들의 집합으로 이루어진 개체(Entity), 그리고 개체 사이에 존재하는 관계(Relation)에 대한 정의를 포함하여, 이들이 지켜야 할 제약 조건을 기술한 것입니다.
    개체(Entity)

        데이터로 표현하려고 하는 객체 (여러 속성들로 구성)

        ER 다이어그램에서 네모로 표현

        ex) 학생, 과목

    속성(Attribute)

        개체가 갖는 속성

        ER 다이어그램에서 원으로 표현

        ex) 홍길동, 이순신, 수학, 영어

    관계(Relation)

        개체와 개체 사이의 연관성

        ER 다이어그램에서 마름모로 표현

        ex) 학생과 과목 간의 “수강”이라는 관계를 가짐
#### 1st. Define entities
#### 2nd. Identify relationships
- 관계의 설정
  - 1:1 관계(일대일) : 관계가 있는 엔티티의 값이 서로 한개의 관계를 갖는것
  - 1:n 관계(일대다) : 관계가 있는 한쪽 엔티티의 하나의 값이 다른 엔티티의 여러 값을 참조하고 있는것
  - n:m 관계(다대다) : 관계가 있는 양쪽 엔티티의 값들이 서로 1:N 관계를 갖는것
#### 3rd. normalize tables to reduce redundancy but denormalize certain fields (e.g., product price in `OrderItems`) for faster queries.




### SQL
- def 
  - SQL (Structured Query Language) is a language used to interact with a relational database—you use it to insert, retrieve, update, and delete data from the database.