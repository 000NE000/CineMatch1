CREATE DATABASE MovieAnalyticsDB;

\c MovieAnalyticsDB

-- Master Reference Tables

CREATE TABLE TriggerMaster (
    TriggerID SERIAL PRIMARY KEY,
    TriggerName VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE InsightMaster (
    InsightID SERIAL PRIMARY KEY,
    InsightDescription VARCHAR(255) NOT NULL UNIQUE
);

-- EmotionMaster: Although emotions are dynamically generated, new entries can be added.
CREATE TABLE EmotionMaster (
    EmotionID SERIAL PRIMARY KEY,
    EmotionName VARCHAR(255) NOT NULL UNIQUE
);

-- Intermediate Table: Trigger to Insight Mapping (predefined relationships)
CREATE TABLE TriggerInsightMapping (
    MappingID SERIAL PRIMARY KEY,
    TriggerID INT NOT NULL REFERENCES TriggerMaster(TriggerID),
    InsightID INT NOT NULL REFERENCES InsightMaster(InsightID),
    UNIQUE (TriggerID, InsightID)
);

-- Movie-Specific Tables

CREATE TABLE Movie (
    MovieID SERIAL PRIMARY KEY,
    Title VARCHAR(255) NOT NULL,
    ReleaseYear INT
);

-- Intermediate Table: Movie to Trigger Mapping
CREATE TABLE MovieTriggers (
    MovieTriggersID SERIAL PRIMARY KEY,
    MovieID INT NOT NULL REFERENCES Movie(MovieID),
    TriggerID INT NOT NULL REFERENCES TriggerMaster(TriggerID),
    UNIQUE (MovieID, TriggerID)
);

-- Intermediate Table: Movie to Emotion Mapping
CREATE TABLE MovieEmotions (
    MovieEmotionsID SERIAL PRIMARY KEY,
    MovieID INT NOT NULL REFERENCES Movie(MovieID),
    -- If using EmotionMaster reference:
    EmotionID INT REFERENCES EmotionMaster(EmotionID),
    -- Optionally store emotion label directly:
    EmotionName VARCHAR(255),
    ConfidenceScore DECIMAL(5,2),
    UNIQUE (MovieID, EmotionID)
    -- Alternatively, if not using EmotionID, you could use:
    -- UNIQUE (MovieID, EmotionName)
);

-- Intermediate Table: Movie to Insight Mapping (for additional/dynamic insights)
CREATE TABLE MovieInsights (
    MovieInsightsID SERIAL PRIMARY KEY,
    MovieID INT NOT NULL REFERENCES Movie(MovieID),
    InsightID INT NOT NULL REFERENCES InsightMaster(InsightID),
    UNIQUE (MovieID, InsightID)
);