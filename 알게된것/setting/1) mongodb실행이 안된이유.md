## 📌 MongoDB 인증 문제 해결 및 관련 개념 정리

### **1️⃣ 문제 상황**
- `mongodb.py` 실행 시 **인증 실패 오류** 발생:
  ```plaintext
  pymongo.errors.OperationFailure: Authentication failed., full error: {'ok': 0.0, 'errmsg': 'Authentication failed.', 'code': 18, 'codeName': 'AuthenticationFailed'}
  ```
- `mongosh -u cinematch1_admin -p 5891 --authenticationDatabase admin` 실행 시 **인증 실패**.
- `MovieNarrativeDB`에 계정이 생성되어 있어 발생한 문제.

---

### **2️⃣ 원인 분석**
#### **1. 사용자가 `MovieNarrativeDB`에서 생성됨**
- MongoDB에서는 사용자가 생성된 데이터베이스에서만 인증 가능.
- 기본적으로 **MongoDB는 `admin` 데이터베이스에서 인증**을 수행함.
- `MovieNarrativeDB`에 사용자가 생성되면, `admin`에서 인증을 시도할 경우 인증 실패.

#### **2. `authenticationDatabase`가 잘못 설정됨**
- `authenticationDatabase=admin`으로 인증을 시도했지만, 사용자는 `MovieNarrativeDB`에 존재.
- 따라서 `authenticationDatabase=MovieNarrativeDB`로 변경해야 하거나, 올바른 데이터베이스에서 사용자 재생성이 필요.
- 
#### **3. `__init__.py` 추가 필요**
- Python이 `config` 폴더를 패키지로 인식하지 못해 `ModuleNotFoundError` 발생.
- 해결 방법: `config/` 폴더에 빈 `__init__.py` 파일 추가.
  ```bash
  touch config/__init__.py
  ```
- 이후 `mongodb.py`에서 올바른 경로로 import 가능:
  ```python
  from config.config import MONGODB_CONFIG
  ```
---

### **3️⃣ 해결 방법**
#### ✅ **1. MongoDB Shell 접속**
```bash
mongosh
```

#### ✅ **2. `MovieNarrativeDB`에 있는 사용자 확인**
```bash
use MovieNarrativeDB
db.getUsers()
```
- 출력된 사용자 목록에서 `cinematch1_admin`이 존재하는지 확인.

#### ✅ **3. `MovieNarrativeDB`에서 사용자 삭제**
```bash
db.dropUser("cinematch1_admin")
```
- 사용자 삭제 후 다시 확인:
  ```bash
  db.getUsers()
  ```
  → `cinematch1_admin`이 삭제되었는지 확인.

#### ✅ **4. `admin`에서 올바른 사용자 생성**
```bash
use admin
db.createUser({
  user: "cinematch1_admin",
  pwd: "5891",
  roles: [{ role: "readWrite", db: "MovieNarrativeDB" }]
})
```
- 이제 `admin`에서 인증할 수 있도록 사용자 생성.

#### ✅ **5. MongoDB 서비스 재시작 (필요 시)**
- macOS (Homebrew 설치 MongoDB):
  ```bash
  brew services restart mongodb-community
  ```
- Linux (systemd 기반):
  ```bash
  sudo systemctl restart mongod
  ```

#### ✅ **6. 인증 테스트**
```bash
mongosh -u cinematch1_admin -p 5891 --authenticationDatabase admin
```
- 정상적으로 로그인되는지 확인.

#### ✅ **7. Python 코드에서 `authSource=admin` 추가**
```python
MONGODB_CONFIG = {
    "uri": "mongodb://cinematch1_admin:5891@localhost:27017/MovieNarrativeDB?authSource=admin",
    "db": "MovieNarrativeDB"
}
```
- `authSource=admin`을 추가하여 올바르게 인증되도록 설정.

---

### **4️⃣ 관련 개념 정리**
#### **1. MongoDB 인증(`authentication`) 개념**
- MongoDB는 **사용자가 생성된 데이터베이스에서만 인증 가능**.
- 기본적으로 **`admin` 데이터베이스에서 인증을 수행**함.
- 사용자가 특정 데이터베이스에서 생성되면, `authSource`를 해당 DB로 지정해야 함.

#### **2. `authenticationDatabase`의 역할**
- `authenticationDatabase`는 **사용자 인증을 수행할 데이터베이스를 지정**하는 옵션.
- 기본값은 `admin`, 하지만 특정 DB에서 사용자가 생성되었다면 해당 DB를 명시해야 함.

#### **3. MongoDB 사용자 관리 명령어 정리**
| 명령어 | 설명 |
|--------|------|
| `use <db>` | 해당 데이터베이스로 이동 |
| `db.getUsers()` | 현재 데이터베이스의 사용자 목록 조회 |
| `db.createUser({...})` | 새로운 사용자 생성 |
| `db.dropUser("user")` | 특정 사용자 삭제 |
| `mongosh -u <user> -p <password> --authenticationDatabase <db>` | 특정 데이터베이스에서 인증 |

---

