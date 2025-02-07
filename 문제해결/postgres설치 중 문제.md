brew services start postgresql📌 1. CS 개념 정리

1️⃣ 환경 변수 (Environment Variables)

✅ 개념
	•	환경 변수는 운영 체제에서 프로그램이 실행될 때 사용할 전역 설정값을 저장하는 시스템 변수야.
	•	보통 터미널에서 export VAR_NAME=value로 설정할 수 있어.

🔹 관련된 문제
	•	locale 설정이 잘못되어 initdb 실행 시 “invalid locale settings” 오류가 발생함.
	•	export LANG=en_US.UTF-8을 설정하여 해결.

2️⃣ 로케일 (Locale)

✅ 개념
	•	운영 체제가 언어, 날짜 형식, 숫자 형식, 문자 인코딩을 관리하는 설정값이야.
	•	locale 명령어로 현재 설정을 확인할 수 있음.

🔹 관련된 문제
	•	initdb -D /opt/homebrew/var/postgresql 실행 시 “invalid locale settings” 오류 발생.
	•	export LANG=en_US.UTF-8 설정을 추가하고, locale-gen 명령어를 실행하여 해결

3️⃣ 프로세스 (Process)

✅ 개념
	•	운영 체제에서 실행 중인 프로그램을 의미하며, 각 프로세스는 PID(Process ID)를 가짐.
	•	ps aux | grep <프로그램명> 또는 pgrep <프로그램명>으로 확인할 수 있음.

🔹 관련된 문제
	•	PostgreSQL을 중지해도 5432 포트에서 계속 프로세스가 살아있는 문제가 발생.
	•	pkill -9 postgres로 프로세스를 강제 종료하여 해결.

4️⃣ 포트 (Port) & 네트워크 소켓 (Socket)

✅ 개념
	•	포트는 네트워크에서 데이터를 주고받는 특정한 논리적 통로이며, PostgreSQL의 기본 포트는 5432.
	•	**Unix 소켓(/tmp/.s.PGSQL.5432)**을 사용하여 로컬 통신이 이루어짐.

🔹 관련된 문제
	•	PostgreSQL 프로세스가 비정상적으로 종료되면서 소켓 파일이 남아 PostgreSQL이 다시 실행되지 않는 문제 발생. 
- 해결 rm -rf /tmp/.s.PGSQL.5432

5️⃣ 신호 (Signal)

✅ 개념
	•	UNIX/Linux에서 프로세스를 제어하기 위해 사용하는 시스템 메시지.
	•	kill -9 <PID>는 **SIGKILL(9)**을 보내 프로세스를 강제 종료함.
	•	pkill -9 <프로세스명>을 사용하면 해당 프로세스를 전부 종료할 수 있음.

🔹 관련된 문제
	•	PostgreSQL이 계속 자동으로 재시작되어 완전히 종료되지 않는 문제 발생.
	•	brew services stop postgresql@16 실행 후, pkill -9 postgres로 강제 종료하여 해결.

6️⃣ 데이터 디렉터리 (Data Directory)

✅ 개념
	•	PostgreSQL은 데이터베이스 정보를 저장할 전용 디렉터리를 가짐.
	•	PostgreSQL 설치 후 반드시 initdb를 실행하여 데이터 디렉터리를 초기화해야 함.

🔹 관련된 문제
	•	PostgreSQL 데이터 디렉터리(/opt/homebrew/var/postgresql or /usr/local/var/postgres)가 손상되거나 올바르게 설정되지 않아 서버가 실행되지 않는 문제 발생.
	•	해결책: 
sudo rm -rf /opt/homebrew/var/postgresql
initdb -D /opt/homebrew/var/postgresql --locale=en_US.UTF-8

7️⃣ 사용자 권한 및 Role (User & Role)

✅ 개념
	•	PostgreSQL은 유닉스의 사용자(User)와 비슷한 개념으로 Role을 사용함.
	•	기본적으로 postgres라는 슈퍼유저(Superuser) 역할이 있어야 함.

🔹 관련된 문제
	•	psql -U postgres 실행 시 “FATAL: role ‘postgres’ does not exist” 오류 발생.
	•	해결책: `createuser -s postgres`

8️⃣ PostgreSQL 서비스 관리 (Homebrew & pg_ctl)

✅ 개념
	•	brew services start postgresql@16: Homebrew 서비스로 PostgreSQL 실행.
	•	pg_ctl -D /path/to/data start: PostgreSQL을 직접 실행.

🔹 관련된 문제
	•	brew services start postgresql@16 실행 후에도 서버가 실행되지 않는 문제 발생.
	•	pg_ctl status -D /opt/homebrew/var/postgresql 실행 시 no server running 오류 발생.
	•	해결책: `pg_ctl -D /opt/homebrew/var/postgresql start`

