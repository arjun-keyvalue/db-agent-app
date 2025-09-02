-- migration_with_seed.sql
-- SQLite3 Migration + Sample Data for Employee Management System Schema (with Work From Home)

PRAGMA foreign_keys = ON;

-----------------------------------------------------
-- 1. Employees
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Employees (
    employee_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name       TEXT NOT NULL,
    last_name        TEXT NOT NULL,
    email            TEXT NOT NULL UNIQUE,
    phone            TEXT,
    address          TEXT,
    password         TEXT NOT NULL,
    hire_date        DATE NOT NULL,
    status           TEXT NOT NULL CHECK(status IN ('Active','Resigned','On Leave')),
    role_id          INTEGER,
    manager_id       INTEGER,
    team_id          INTEGER,
    FOREIGN KEY (role_id) REFERENCES Roles(role_id) ON DELETE SET NULL,
    FOREIGN KEY (manager_id) REFERENCES Employees(employee_id) ON DELETE SET NULL,
    FOREIGN KEY (team_id) REFERENCES Teams(team_id) ON DELETE SET NULL
);

-----------------------------------------------------
-- 2. Departments
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Departments (
    department_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL UNIQUE,
    head_id          INTEGER,
    FOREIGN KEY (head_id) REFERENCES Employees(employee_id) ON DELETE SET NULL
);

-----------------------------------------------------
-- 3. Teams
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Teams (
    team_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    department_id    INTEGER NOT NULL,
    lead_id          INTEGER,
    FOREIGN KEY (department_id) REFERENCES Departments(department_id) ON DELETE CASCADE,
    FOREIGN KEY (lead_id) REFERENCES Employees(employee_id) ON DELETE SET NULL
);

-----------------------------------------------------
-- 4. Roles
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Roles (
    role_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT NOT NULL,
    description      TEXT,
    level            TEXT CHECK(level IN ('Junior','Mid','Senior','Lead')),
    salary_grade     TEXT
);

-----------------------------------------------------
-- 5. Skills
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Skills (
    skill_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    category         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Employee_Skills (
    employee_id      INTEGER NOT NULL,
    skill_id         INTEGER NOT NULL,
    proficiency_level TEXT NOT NULL CHECK(proficiency_level IN ('Beginner','Intermediate','Expert')),
    last_used        DATE,
    PRIMARY KEY (employee_id, skill_id),
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES Skills(skill_id) ON DELETE CASCADE
);

-----------------------------------------------------
-- 6. Projects
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Projects (
    project_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    description      TEXT,
    start_date       DATE,
    end_date         DATE,
    status           TEXT CHECK(status IN ('Ongoing','Completed','On Hold')),
    product_owner_id INTEGER,
    required_skill_id INTEGER,
    FOREIGN KEY (product_owner_id) REFERENCES Employees(employee_id) ON DELETE SET NULL,
    FOREIGN KEY (required_skill_id) REFERENCES Skills(skill_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS Employee_Project (
    employee_id      INTEGER NOT NULL,
    project_id       INTEGER NOT NULL,
    role_in_project  TEXT NOT NULL,
    allocation_percentage INTEGER,
    PRIMARY KEY (employee_id, project_id),
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES Projects(project_id) ON DELETE CASCADE
);

-----------------------------------------------------
-- 7. Performance Reviews
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Performance_Reviews (
    review_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id      INTEGER NOT NULL,
    reviewer_id      INTEGER,
    review_cycle     TEXT NOT NULL,
    rating           INTEGER CHECK(rating BETWEEN 1 AND 5),
    comments         TEXT,
    goals            TEXT,
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
    FOREIGN KEY (reviewer_id) REFERENCES Employees(employee_id) ON DELETE SET NULL
);

-----------------------------------------------------
-- 8. Training Programs
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Training_Programs (
    training_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT NOT NULL,
    description      TEXT,
    start_date       DATE,
    end_date         DATE,
    instructor_id    INTEGER,
    FOREIGN KEY (instructor_id) REFERENCES Employees(employee_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS Employee_Training (
    employee_id      INTEGER NOT NULL,
    training_id      INTEGER NOT NULL,
    completion_status TEXT CHECK(completion_status IN ('Not Started','In Progress','Completed')),
    PRIMARY KEY (employee_id, training_id),
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
    FOREIGN KEY (training_id) REFERENCES Training_Programs(training_id) ON DELETE CASCADE
);

-----------------------------------------------------
-- 9. Leave Requests
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Leave_Requests (
    leave_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id      INTEGER NOT NULL,
    leave_type       TEXT NOT NULL,
    start_date       DATE NOT NULL,
    end_date         DATE NOT NULL,
    status           TEXT CHECK(status IN ('Pending','Approved','Rejected')),
    approved_by      INTEGER,
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
    FOREIGN KEY (approved_by) REFERENCES Employees(employee_id) ON DELETE SET NULL
);

-----------------------------------------------------
-- 10. Payroll
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Payroll (
    payroll_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id      INTEGER NOT NULL,
    base_salary      REAL NOT NULL,
    bonus            REAL,
    stock_options    REAL,
    pay_date         DATE NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE
);

-----------------------------------------------------
-- 11. Work From Home
-----------------------------------------------------
CREATE TABLE IF NOT EXISTS Work_From_Home (
    wfh_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id      INTEGER NOT NULL,
    wfh_date         DATE NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE
);

-----------------------------------------------------
-- SAMPLE DATA
-----------------------------------------------------

-- Roles
INSERT INTO Roles (title, description, level, salary_grade) VALUES
('Software Engineer', 'Responsible for developing software', 'Junior', 'S1'),
('QA Engineer', 'Responsible for testing', 'Mid', 'S2'),
('Team Lead', 'Leads a small team', 'Lead', 'S3'),
('Product Manager', 'Owns product strategy', 'Senior', 'S4');

-- Skills
INSERT INTO Skills (name, category) VALUES
('Python', 'Programming Language'),
('React', 'Framework'),
('Kubernetes', 'Cloud'),
('SQL', 'Tool');

-- Departments
INSERT INTO Departments (name) VALUES
('Engineering'),
('Product'),
('HR');

-- Teams
INSERT INTO Teams (name, department_id) VALUES
('Backend', 1),
('QA', 1),
('Product Strategy', 2);

-- Employees
INSERT INTO Employees (first_name, last_name, email, phone, address, password, hire_date, status, role_id, manager_id, team_id) VALUES
('Alice', 'Johnson', 'alice.johnson@example.com', '555-1111', '123 Main St, City, State 12345', 'hashed_password_123', '2022-01-15', 'Active', 1, NULL, 1),
('Bob', 'Smith', 'bob.smith@example.com', '555-2222', '456 Oak Ave, Town, State 67890', 'hashed_password_456', '2021-05-10', 'Active', 3, NULL, 1),
('Charlie', 'Davis', 'charlie.davis@example.com', '555-3333', '789 Pine Rd, Village, State 11111', 'hashed_password_789', '2023-03-20', 'On Leave', 2, 2, 2),
('Diana', 'Moore', 'diana.moore@example.com', '555-4444', '321 Elm St, Borough, State 22222', 'hashed_password_321', '2020-07-01', 'Active', 4, NULL, 3),
('Evan', 'Lee', 'evan.lee@example.com', '555-5555', '654 Maple Dr, District, State 33333', 'hashed_password_654', '2019-09-25', 'Resigned', 1, 2, 1);

-- Update heads & leads
UPDATE Departments SET head_id = 2 WHERE department_id = 1;
UPDATE Departments SET head_id = 4 WHERE department_id = 2;
UPDATE Teams SET lead_id = 2 WHERE team_id = 1;
UPDATE Teams SET lead_id = 3 WHERE team_id = 2;
UPDATE Teams SET lead_id = 4 WHERE team_id = 3;

-- Employee Skills
INSERT INTO Employee_Skills (employee_id, skill_id, proficiency_level, last_used) VALUES
(1, 1, 'Expert', '2024-12-01'),
(1, 4, 'Intermediate', '2024-11-15'),
(2, 1, 'Intermediate', '2024-10-20'),
(3, 2, 'Expert', '2024-08-10'),
(4, 3, 'Beginner', '2024-09-01');

-- Projects
INSERT INTO Projects (name, description, start_date, end_date, status, product_owner_id, required_skill_id) VALUES
('Internal API', 'Backend API for internal tools', '2023-01-01', '2023-12-31', 'Completed', 2, 1),
('Frontend Revamp', 'New React-based UI', '2024-01-15', NULL, 'Ongoing', 4, 2),
('Cloud Migration', 'Move services to Kubernetes', '2024-05-01', NULL, 'On Hold', 2, 3);

-- Employee Projects
INSERT INTO Employee_Project (employee_id, project_id, role_in_project, allocation_percentage) VALUES
(1, 1, 'Developer', 100),
(2, 1, 'Architect', 50),
(3, 2, 'Tester', 75),
(4, 2, 'Product Owner', 100),
(2, 3, 'Lead Developer', 50);

-- Performance Reviews
INSERT INTO Performance_Reviews (employee_id, reviewer_id, review_cycle, rating, comments, goals) VALUES
(1, 2, 'Q1-2024', 4, 'Strong technical skills, needs more leadership experience', '{"goal":"Mentor junior devs"}'),
(3, 2, 'Q2-2024', 3, 'Good tester, but attendance issues due to leave', '{"goal":"Improve consistency"}'),
(4, NULL, 'Q3-2024', 5, 'Excellent strategic vision', '{"goal":"Expand product line"}');

-- Training Programs
INSERT INTO Training_Programs (title, description, start_date, end_date, instructor_id) VALUES
('Advanced Python', 'Deep dive into Python for backend devs', '2024-06-01', '2024-06-15', 2),
('Agile Practices', 'Training on Agile methodologies', '2024-07-10', '2024-07-20', NULL);

-- Employee Training
INSERT INTO Employee_Training (employee_id, training_id, completion_status) VALUES
(1, 1, 'Completed'),
(2, 1, 'Completed'),
(3, 2, 'In Progress'),
(4, 2, 'Not Started');

-- Leave Requests
INSERT INTO Leave_Requests (employee_id, leave_type, start_date, end_date, status, approved_by) VALUES
(1, 'Vacation', '2024-12-20', '2024-12-30', 'Approved', 2),
(3, 'Sick', '2024-09-01', '2024-09-10', 'Pending', 2);

-- Payroll
INSERT INTO Payroll (employee_id, base_salary, bonus, stock_options, pay_date) VALUES
(1, 60000, 5000, 1000, '2024-12-31'),
(2, 90000, 10000, 5000, '2024-12-31'),
(3, 70000, 4000, 2000, '2024-12-31'),
(4, 120000, 20000, 10000, '2024-12-31');

-- Work From Home
INSERT INTO Work_From_Home (employee_id, wfh_date) VALUES
(1, '2024-11-05'),
(2, '2024-11-05'),
(3, '2024-08-15'),
(4, '2024-10-20'),
(1, '2024-12-10');
