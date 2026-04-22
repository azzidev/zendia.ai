package datasource

import (
	"context"
	"fmt"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type MongoStore struct {
	client *mongo.Client
	db     *mongo.Database
}

type Document struct {
	Title string `bson:"title" json:"title"`
	Text  string `bson:"text" json:"text"`
	Chars int    `bson:"chars" json:"chars"`
}

type DialogueDoc struct {
	User string `bson:"user" json:"user"`
	Bot  string `bson:"bot" json:"bot"`
}

func NewMongoStore(uri, dbName string) (*MongoStore, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		return nil, fmt.Errorf("erro ao conectar no mongo: %w", err)
	}

	if err := client.Ping(ctx, nil); err != nil {
		return nil, fmt.Errorf("mongo não respondeu: %w", err)
	}

	fmt.Println("✅ Conectado ao MongoDB")
	return &MongoStore{
		client: client,
		db:     client.Database(dbName),
	}, nil
}

func (m *MongoStore) Collection() *mongo.Collection {
	return m.db.Collection("articles")
}

func (m *MongoStore) DialogueCollection() *mongo.Collection {
	return m.db.Collection("dialogues")
}

func (m *MongoStore) InsertDialogueBatch(docs []DialogueDoc) error {
	if len(docs) == 0 {
		return nil
	}
	col := m.DialogueCollection()
	batch := make([]interface{}, len(docs))
	for i, d := range docs {
		batch[i] = d
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err := col.InsertMany(ctx, batch)
	return err
}

func (m *MongoStore) DialogueCount() (int64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return m.DialogueCollection().CountDocuments(ctx, bson.M{})
}

func (m *MongoStore) GetDialogues(maxDocs int) ([]DialogueDoc, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	opts := options.Find().SetLimit(int64(maxDocs))
	cursor, err := m.DialogueCollection().Find(ctx, bson.M{}, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var docs []DialogueDoc
	if err := cursor.All(ctx, &docs); err != nil {
		return nil, err
	}
	return docs, nil
}

func (m *MongoStore) InsertBatch(docs []Document) error {
	if len(docs) == 0 {
		return nil
	}
	col := m.Collection()
	batch := make([]interface{}, len(docs))
	for i, d := range docs {
		batch[i] = d
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err := col.InsertMany(ctx, batch)
	return err
}

func (m *MongoStore) Count() (int64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return m.Collection().CountDocuments(ctx, bson.M{})
}

func (m *MongoStore) GetBatch(skip, limit int) ([]Document, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	opts := options.Find().SetSkip(int64(skip)).SetLimit(int64(limit))
	cursor, err := m.Collection().Find(ctx, bson.M{}, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var docs []Document
	if err := cursor.All(ctx, &docs); err != nil {
		return nil, err
	}
	return docs, nil
}

// GetAllText retorna todo o texto concatenado (pra treinar o tokenizer)
func (m *MongoStore) GetAllText(maxDocs int) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	opts := options.Find().SetLimit(int64(maxDocs)).SetProjection(bson.M{"text": 1})
	cursor, err := m.Collection().Find(ctx, bson.M{}, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var texts []string
	for cursor.Next(ctx) {
		var doc Document
		if err := cursor.Decode(&doc); err == nil && doc.Text != "" {
			texts = append(texts, doc.Text)
		}
	}
	return texts, nil
}

func (m *MongoStore) Drop() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return m.Collection().Drop(ctx)
}

func (m *MongoStore) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	m.client.Disconnect(ctx)
}
