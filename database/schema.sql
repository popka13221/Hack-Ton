-- PostgreSQL schema for the sentiment analysis web service.
-- Apply with: psql -U <user> -d <database> -f database/schema.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    title TEXT
);

CREATE TABLE IF NOT EXISTS batches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_name TEXT,
    purpose TEXT NOT NULL CHECK (purpose IN ('predict', 'score')),
    status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('processing', 'completed', 'failed')),
    total_rows INTEGER,
    class_counts JSONB,
    macro_f1 NUMERIC(5, 4),
    f1_per_class JSONB,
    support JSONB,
    output_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS reviews (
    id BIGSERIAL PRIMARY KEY,
    batch_id UUID REFERENCES batches(id) ON DELETE SET NULL,
    row_idx INTEGER,
    src_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    input_text TEXT NOT NULL,
    true_label SMALLINT CHECK (true_label BETWEEN 0 AND 2),
    predicted_label SMALLINT CHECK (predicted_label BETWEEN 0 AND 2),
    predicted_scores JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reviews_batch ON reviews(batch_id);
CREATE INDEX IF NOT EXISTS idx_reviews_src ON reviews(src_id);
CREATE INDEX IF NOT EXISTS idx_reviews_predicted_label ON reviews(predicted_label);

CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_version TEXT NOT NULL,
    dataset TEXT,
    macro_f1 NUMERIC(5, 4) NOT NULL,
    f1_per_class JSONB,
    support JSONB,
    params JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO sources (code, title) VALUES
    ('app_store', 'App Store'),
    ('mos_ru', 'mos.ru'),
    ('vk', 'VK')
ON CONFLICT (code) DO NOTHING;
