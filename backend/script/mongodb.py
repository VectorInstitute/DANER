import mongoengine as db

db.connect(host="mongodb://127.0.0.1:27017/nera_db")

ANNOTATION_QUALITY = (('H', 'High'), ('M', 'Medium'), ('L', 'Low'))


class User(db.Document):
    username = db.StringField(max_length=30, required=True, unique=True)
    password = db.StringField(default='', max_length=30)
    created_at = db.DateTimeField(auto_now_add=True)


class Dataset(db.Document):
    name = db.StringField(max_length=30, required=True, unique=True)
    length = db.IntField(default=0)
    created_by = db.ReferenceField(User)
    created_at = db.DateTimeField(auto_now_add=True)


class Example(db.Document):
    # Delete all examples when the dataset is deleted
    dataset = db.ReferenceField(Dataset, reverse_delete_rule=db.CASCADE)
    hash = db.StringField(max_length=256, unique=True)

    meta = {'allow_inheritance': True}


class Token(db.EmbeddedDocument):
    id = db.IntField(required=True)
    text = db.StringField(required=True, max_length=30)
    start = db.IntField(required=True)
    end = db.IntField(required=True)


class Span(db.EmbeddedDocument):
    id = db.IntField(required=True)
    text = db.StringField(required=True, max_length=30)
    start = db.IntField(required=True)
    end = db.IntField(required=True)
    token_start = db.IntField(required=True)
    token_end = db.IntField(required=True)
    label = db.StringField(required=True, max_length=30)


class Annotation(db.EmbeddedDocument):
    id = db.IntField(required=True)
    quality = db.StringField(
        required=True, choices=ANNOTATION_QUALITY, max_length=30)
    annotator = db.ReferenceField(User)
    spans = db.EmbeddedDocumentListField(Span)


class TextExample(Example):
    text = db.StringField(required=True)
    tokens = db.EmbeddedDocumentListField(Token)
    annotations = db.EmbeddedDocumentListField(Annotation)
    prediction = db.EmbeddedDocumentField(Annotation)

    meta = {
        # 100000 Examples, 20 MB
        'max_documents': 100000,
        'max_size': 20000000,
        'indexes': [
            {'fields': ['$text'],
             'default_language': 'english',
             'weights': {'text': 10}}
        ]
    }
