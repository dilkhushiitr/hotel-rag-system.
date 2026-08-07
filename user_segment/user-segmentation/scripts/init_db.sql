-- ─────────────────────────────────────────────────────────────────────────────
-- init_db.sql
-- Initialises all tables in the segmentation_db PostgreSQL database.
-- Runs automatically on first container start via docker-entrypoint-initdb.d/
-- ─────────────────────────────────────────────────────────────────────────────

-- ── 1. Users ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id      VARCHAR(20)  PRIMARY KEY,
    created_at   TIMESTAMP    NOT NULL,
    country      VARCHAR(5),
    device_type  VARCHAR(20),
    age_bucket   VARCHAR(10),
    referral_src VARCHAR(30)
);

-- ── 2. Sessions ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id         VARCHAR(20)  PRIMARY KEY,
    user_id            VARCHAR(20)  REFERENCES users(user_id),
    session_start      TIMESTAMP    NOT NULL,
    duration_seconds   INTEGER,
    pages_viewed       INTEGER,
    platform           VARCHAR(20)
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(session_start);

-- ── 3. Ad Events ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ad_events (
    event_id     VARCHAR(20)  PRIMARY KEY,
    user_id      VARCHAR(20)  REFERENCES users(user_id),
    ad_id        VARCHAR(20),
    event_type   VARCHAR(20)  NOT NULL,    -- 'impression' or 'click'
    timestamp    TIMESTAMP    NOT NULL,
    ad_category  VARCHAR(30)
);
CREATE INDEX IF NOT EXISTS idx_ad_events_user ON ad_events(user_id);
CREATE INDEX IF NOT EXISTS idx_ad_events_ts   ON ad_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_ad_events_type ON ad_events(event_type);

-- ── 4. Wallet ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wallet (
    user_id          VARCHAR(20)  PRIMARY KEY REFERENCES users(user_id),
    total_earnings   NUMERIC(12,2) DEFAULT 0,
    total_redeemed   NUMERIC(12,2) DEFAULT 0,
    wallet_balance   NUMERIC(12,2) DEFAULT 0,
    last_redemption  TIMESTAMP
);

-- ── 5. Transactions ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    txn_id      VARCHAR(20)  PRIMARY KEY,
    user_id     VARCHAR(20)  REFERENCES users(user_id),
    amount      NUMERIC(12,2),
    status      VARCHAR(20),              -- 'success', 'failed', 'pending'
    txn_type    VARCHAR(30),              -- 'purchase', 'withdrawal', 'deposit'
    created_at  TIMESTAMP    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_txn_user ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(created_at);

-- ── 6. User Segments (output table) ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_segments (
    user_id             VARCHAR(20)  PRIMARY KEY,
    kmeans_cluster      INTEGER,
    dbscan_cluster      INTEGER,
    segment_label       VARCHAR(60),
    is_anomaly          SMALLINT     DEFAULT 0,
    engagement_score    NUMERIC(6,2),
    monetization_score  NUMERIC(6,2),
    churn_risk_score    NUMERIC(6,2),
    last_updated        TIMESTAMP    NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_seg_label   ON user_segments(segment_label);
CREATE INDEX IF NOT EXISTS idx_seg_anomaly ON user_segments(is_anomaly);

-- Done
SELECT 'Database initialised successfully.' AS status;
