-- migration_with_seed.sql
-- PostgreSQL Migration + Sample Data for Employee Management System Schema (with Work From Home)

-----------------------------------------------------
-- 1. Create all tables without foreign key constraints
-----------------------------------------------------

-- Employees
CREATE TABLE IF NOT EXISTS Employees (
    employee_id      SERIAL PRIMARY KEY,
    first_name       VARCHAR(100) NOT NULL,
    last_name        VARCHAR(100) NOT NULL,
    email            VARCHAR(255) NOT NULL UNIQUE,
    phone            VARCHAR(20),
    address          TEXT,
    password         VARCHAR(255) NOT NULL,
    hire_date        DATE NOT NULL,
    status           VARCHAR(20) NOT NULL CHECK(status IN ('Active','Resigned','On Leave')),
    role_id          INTEGER,
    manager_id       INTEGER,
    team_id          INTEGER
);

-- Departments
CREATE TABLE IF NOT EXISTS Departments (
    department_id    SERIAL PRIMARY KEY,
    name             VARCHAR(100) NOT NULL UNIQUE,
    head_id          INTEGER
);

-- Teams
CREATE TABLE IF NOT EXISTS Teams (
    team_id          SERIAL PRIMARY KEY,
    name             VARCHAR(100) NOT NULL,
    department_id    INTEGER NOT NULL,
    lead_id          INTEGER
);

-- Roles
CREATE TABLE IF NOT EXISTS Roles (
    role_id          SERIAL PRIMARY KEY,
    title            VARCHAR(100) NOT NULL,
    description      TEXT,
    level            VARCHAR(20) CHECK(level IN ('Junior','Mid','Senior','Lead')),
    salary_grade     VARCHAR(10)
);

-- Skills
CREATE TABLE IF NOT EXISTS Skills (
    skill_id         SERIAL PRIMARY KEY,
    name             VARCHAR(100) NOT NULL,
    category         VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS Employee_Skills (
    employee_id      INTEGER NOT NULL,
    skill_id         INTEGER NOT NULL,
    proficiency_level VARCHAR(20) NOT NULL CHECK(proficiency_level IN ('Beginner','Intermediate','Expert')),
    last_used        DATE,
    PRIMARY KEY (employee_id, skill_id)
);

-- Projects
CREATE TABLE IF NOT EXISTS Projects (
    project_id       SERIAL PRIMARY KEY,
    name             VARCHAR(100) NOT NULL,
    description      TEXT,
    start_date       DATE,
    end_date         DATE,
    status           VARCHAR(20) CHECK(status IN ('Ongoing','Completed','On Hold')),
    product_owner_id INTEGER,
    required_skill_id INTEGER
);

CREATE TABLE IF NOT EXISTS Employee_Project (
    employee_id      INTEGER NOT NULL,
    project_id       INTEGER NOT NULL,
    role_in_project  VARCHAR(100) NOT NULL,
    allocation_percentage INTEGER,
    PRIMARY KEY (employee_id, project_id)
);

-- Performance Reviews
CREATE TABLE IF NOT EXISTS Performance_Reviews (
    review_id        SERIAL PRIMARY KEY,
    employee_id      INTEGER NOT NULL,
    reviewer_id      INTEGER,
    review_cycle     VARCHAR(20) NOT NULL,
    rating           INTEGER CHECK(rating BETWEEN 1 AND 5),
    comments         TEXT,
    goals            TEXT
);

-- Training Programs
CREATE TABLE IF NOT EXISTS Training_Programs (
    training_id      SERIAL PRIMARY KEY,
    title            VARCHAR(100) NOT NULL,
    description      TEXT,
    start_date       DATE,
    end_date         DATE,
    instructor_id    INTEGER
);

CREATE TABLE IF NOT EXISTS Employee_Training (
    employee_id      INTEGER NOT NULL,
    training_id      INTEGER NOT NULL,
    completion_status VARCHAR(20) CHECK(completion_status IN ('Not Started','In Progress','Completed')),
    PRIMARY KEY (employee_id, training_id)
);

-- Leave Requests
CREATE TABLE IF NOT EXISTS Leave_Requests (
    leave_id         SERIAL PRIMARY KEY,
    employee_id      INTEGER NOT NULL,
    leave_type       VARCHAR(50) NOT NULL,
    start_date       DATE NOT NULL,
    end_date         DATE NOT NULL,
    status           VARCHAR(20) CHECK(status IN ('Pending','Approved','Rejected')),
    approved_by      INTEGER
);

-- Payroll
CREATE TABLE IF NOT EXISTS Payroll (
    payroll_id       SERIAL PRIMARY KEY,
    employee_id      INTEGER NOT NULL,
    base_salary      NUMERIC(10,2) NOT NULL,
    bonus            NUMERIC(10,2),
    stock_options    NUMERIC(10,2),
    pay_date         DATE NOT NULL
);

-- Work From Home
CREATE TABLE IF NOT EXISTS Work_From_Home (
    wfh_id           SERIAL PRIMARY KEY,
    employee_id      INTEGER NOT NULL,
    wfh_date         DATE NOT NULL
);

-----------------------------------------------------
-- 2. Add all foreign key constraints
-----------------------------------------------------

-- Employees foreign keys
ALTER TABLE Employees 
ADD CONSTRAINT fk_employees_role FOREIGN KEY (role_id) REFERENCES Roles(role_id) ON DELETE SET NULL,
ADD CONSTRAINT fk_employees_manager FOREIGN KEY (manager_id) REFERENCES Employees(employee_id) ON DELETE SET NULL,
ADD CONSTRAINT fk_employees_team FOREIGN KEY (team_id) REFERENCES Teams(team_id) ON DELETE SET NULL;

-- Departments foreign keys
ALTER TABLE Departments 
ADD CONSTRAINT fk_departments_head FOREIGN KEY (head_id) REFERENCES Employees(employee_id) ON DELETE SET NULL;

-- Teams foreign keys
ALTER TABLE Teams 
ADD CONSTRAINT fk_teams_department FOREIGN KEY (department_id) REFERENCES Departments(department_id) ON DELETE CASCADE,
ADD CONSTRAINT fk_teams_lead FOREIGN KEY (lead_id) REFERENCES Employees(employee_id) ON DELETE SET NULL;

-- Employee_Skills foreign keys
ALTER TABLE Employee_Skills 
ADD CONSTRAINT fk_employee_skills_employee FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
ADD CONSTRAINT fk_employee_skills_skill FOREIGN KEY (skill_id) REFERENCES Skills(skill_id) ON DELETE CASCADE;

-- Projects foreign keys
ALTER TABLE Projects 
ADD CONSTRAINT fk_projects_product_owner FOREIGN KEY (product_owner_id) REFERENCES Employees(employee_id) ON DELETE SET NULL,
ADD CONSTRAINT fk_projects_required_skill FOREIGN KEY (required_skill_id) REFERENCES Skills(skill_id) ON DELETE SET NULL;

-- Employee_Project foreign keys
ALTER TABLE Employee_Project 
ADD CONSTRAINT fk_employee_project_employee FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
ADD CONSTRAINT fk_employee_project_project FOREIGN KEY (project_id) REFERENCES Projects(project_id) ON DELETE CASCADE;

-- Performance_Reviews foreign keys
ALTER TABLE Performance_Reviews 
ADD CONSTRAINT fk_performance_reviews_employee FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
ADD CONSTRAINT fk_performance_reviews_reviewer FOREIGN KEY (reviewer_id) REFERENCES Employees(employee_id) ON DELETE SET NULL;

-- Training_Programs foreign keys
ALTER TABLE Training_Programs 
ADD CONSTRAINT fk_training_programs_instructor FOREIGN KEY (instructor_id) REFERENCES Employees(employee_id) ON DELETE SET NULL;

-- Employee_Training foreign keys
ALTER TABLE Employee_Training 
ADD CONSTRAINT fk_employee_training_employee FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
ADD CONSTRAINT fk_employee_training_training FOREIGN KEY (training_id) REFERENCES Training_Programs(training_id) ON DELETE CASCADE;

-- Leave_Requests foreign keys
ALTER TABLE Leave_Requests 
ADD CONSTRAINT fk_leave_requests_employee FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
ADD CONSTRAINT fk_leave_requests_approved_by FOREIGN KEY (approved_by) REFERENCES Employees(employee_id) ON DELETE SET NULL;

-- Payroll foreign keys
ALTER TABLE Payroll 
ADD CONSTRAINT fk_payroll_employee FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE;

-- Work_From_Home foreign keys
ALTER TABLE Work_From_Home 
ADD CONSTRAINT fk_work_from_home_employee FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE;

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
(1, 60000.00, 5000.00, 1000.00, '2024-12-31'),
(2, 90000.00, 10000.00, 5000.00, '2024-12-31'),
(3, 70000.00, 4000.00, 2000.00, '2024-12-31'),
(4, 120000.00, 20000.00, 10000.00, '2024-12-31');

-- Work From Home
INSERT INTO Work_From_Home (employee_id, wfh_date) VALUES
(1, '2024-11-05'),
(2, '2024-11-05'),
(3, '2024-08-15'),
(4, '2024-10-20'),
(1, '2024-12-10');
