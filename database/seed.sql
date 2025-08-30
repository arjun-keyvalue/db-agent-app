-- Library System Database Schema and Seed Data
-- This script will only create tables and insert data if they don't already exist

-- Create users table if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    phone_number VARCHAR(20),
    date_of_birth DATE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create books table if it doesn't exist
CREATE TABLE IF NOT EXISTS books (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    isbn VARCHAR(13) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(100) NOT NULL,
    genre VARCHAR(50),
    publication_year INTEGER,
    total_copies INTEGER DEFAULT 1,
    available_copies INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create book_loans table if it doesn't exist
CREATE TABLE IF NOT EXISTS book_loans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    book_id UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    loan_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    due_date TIMESTAMP WITH TIME ZONE NOT NULL,
    return_date TIMESTAMP WITH TIME ZONE,
    is_overdue BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create book_reviews table if it doesn't exist
CREATE TABLE IF NOT EXISTS book_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    book_id UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, book_id)
);



-- Insert seed data only if tables are empty
DO $$
BEGIN
    -- Insert users only if the table is empty
    IF (SELECT COUNT(*) FROM users) = 0 THEN
        INSERT INTO users (username, email, password, full_name, phone_number, date_of_birth) VALUES
        ('john_doe', 'john.doe@email.com', 'password123', 'John Doe', '+1234567890', '1990-05-15'),
        ('jane_smith', 'jane.smith@email.com', 'password123', 'Jane Smith', '+1234567891', '1988-12-03'),
        ('bob_wilson', 'bob.wilson@email.com', 'password123', 'Bob Wilson', '+1234567892', '1995-08-22'),
        ('alice_brown', 'alice.brown@email.com', 'password123', 'Alice Brown', '+1234567893', '1992-03-10'),
        ('charlie_davis', 'charlie.davis@email.com', 'password123', 'Charlie Davis', '+1234567894', '1987-11-28');
        
        RAISE NOTICE 'Inserted 5 users';
    ELSE
        RAISE NOTICE 'Users table already has data, skipping user insertion';
    END IF;

    -- Insert books only if the table is empty
    IF (SELECT COUNT(*) FROM books) = 0 THEN
        INSERT INTO books (isbn, title, author, genre, publication_year, total_copies, available_copies) VALUES
        ('9780061120084', 'To Kill a Mockingbird', 'Harper Lee', 'Fiction', 1960, 3, 2),
        ('9780743273565', 'The Great Gatsby', 'F. Scott Fitzgerald', 'Fiction', 1925, 2, 1),
        ('9780140283334', '1984', 'George Orwell', 'Dystopian', 1949, 4, 3),
        ('9780618640157', 'The Lord of the Rings', 'J.R.R. Tolkien', 'Fantasy', 1954, 2, 1),
        ('9780316769488', 'The Catcher in the Rye', 'J.D. Salinger', 'Fiction', 1951, 3, 2),
        ('9780062315007', 'The Alchemist', 'Paulo Coelho', 'Fiction', 1988, 2, 2),
        ('9780547928210', 'The Hobbit', 'J.R.R. Tolkien', 'Fantasy', 1937, 3, 2),
        ('9780743477106', 'Romeo and Juliet', 'William Shakespeare', 'Drama', 1597, 2, 1),
        ('9780141439518', 'Pride and Prejudice', 'Jane Austen', 'Romance', 1813, 3, 2);
        
        RAISE NOTICE 'Inserted 9 books';
    ELSE
        RAISE NOTICE 'Books table already has data, skipping book insertion';
    END IF;

    -- Insert book loans only if the table is empty
    IF (SELECT COUNT(*) FROM book_loans) = 0 THEN
        INSERT INTO book_loans (user_id, book_id, loan_date, due_date, return_date, is_overdue) VALUES
        ((SELECT id FROM users WHERE username = 'john_doe'), (SELECT id FROM books WHERE isbn = '9780061120084'), '2024-01-15', '2024-02-15', NULL, false),
        ((SELECT id FROM users WHERE username = 'jane_smith'), (SELECT id FROM books WHERE isbn = '9780743273565'), '2024-01-10', '2024-02-10', NULL, false),
        ((SELECT id FROM users WHERE username = 'bob_wilson'), (SELECT id FROM books WHERE isbn = '9780140283334'), '2024-01-05', '2024-02-05', NULL, true),
        ((SELECT id FROM users WHERE username = 'alice_brown'), (SELECT id FROM books WHERE isbn = '9780618640157'), '2024-01-20', '2024-02-20', NULL, false),
        ((SELECT id FROM users WHERE username = 'charlie_davis'), (SELECT id FROM books WHERE isbn = '9780316769488'), '2024-01-12', '2024-02-12', '2024-01-25', false);
        
        RAISE NOTICE 'Inserted 5 book loans';
    ELSE
        RAISE NOTICE 'Book loans table already has data, skipping loan insertion';
    END IF;

    -- Insert book reviews only if the table is empty
    IF (SELECT COUNT(*) FROM book_reviews) = 0 THEN
        INSERT INTO book_reviews (user_id, book_id, rating, review_text) VALUES
        ((SELECT id FROM users WHERE username = 'john_doe'), (SELECT id FROM books WHERE isbn = '9780061120084'), 5, 'A masterpiece of American literature. Highly recommended!'),
        ((SELECT id FROM users WHERE username = 'jane_smith'), (SELECT id FROM books WHERE isbn = '9780743273565'), 4, 'Beautiful prose and compelling story. The Jazz Age comes alive.'),
        ((SELECT id FROM users WHERE username = 'bob_wilson'), (SELECT id FROM books WHERE isbn = '9780140283334'), 5, 'Disturbing but essential reading. Still relevant today.'),
        ((SELECT id FROM users WHERE username = 'alice_brown'), (SELECT id FROM books WHERE isbn = '9780618640157'), 5, 'Epic fantasy at its finest. Tolkien is unmatched.'),
        ((SELECT id FROM users WHERE username = 'charlie_davis'), (SELECT id FROM books WHERE isbn = '9780316769488'), 3, 'Interesting perspective on teenage angst, but dated.');
        
        RAISE NOTICE 'Inserted 5 book reviews';
    ELSE
        RAISE NOTICE 'Book reviews table already has data, skipping review insertion';
    END IF;
END $$;

-- Update available copies based on loans (only if we have loans)
UPDATE books SET available_copies = total_copies - (
    SELECT COALESCE(COUNT(*), 0) FROM book_loans 
    WHERE book_id = books.id AND return_date IS NULL
) WHERE EXISTS (SELECT 1 FROM book_loans);



SELECT 'Database initialization completed successfully!' as status;
