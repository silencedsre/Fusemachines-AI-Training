import json
from flask import Flask, request, jsonify
from flask_mongoengine import MongoEngine

app = Flask(__name__)
app.config["MONGODB_SETTINGS"] = {
    "db": "fusemachines_ai_training",
    "host": "localhost",
    "port": 27017,
}
db = MongoEngine()
db.init_app(app)


class User(db.Document):
    name = db.StringField()
    email = db.StringField()

    def to_json(self):
        return {"name": self.name, "email": self.email}


@app.route("/", methods=["GET"])
def query_records():
    user = User.objects()
    return jsonify({"data": user})


@app.route("/", methods=["POST"])
def update_record():
    record = json.loads(request.data)
    print(record)
    user = User(name=record["name"], email=record["email"])
    user.save()
    return jsonify({"success": "success"})


# TODO put and delete
# @app.route('/', methods=['PUT'])
# def create_record():
#     record = json.loads(request.data)
#     user = User(name=record['name'],
#                 email=record['email'])
#     user.save()
#     return jsonify(user.to_json())
#
# @app.route('/', methods=['DELETE'])
# def delete_record():
#     record = json.loads(request.data)
#     user = User.objects(name=record['name']).first()
#     if not user:
#         return jsonify({'error': 'data not found'})
#     else:
#         user.delete()
#     return jsonify(user.to_json())

if __name__ == "__main__":
    app.run(debug=True)
