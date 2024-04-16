import 'dart:developer';

import 'package:mongo_dart/mongo_dart.dart';

class MongoDB {
  static var db;
  static String mongoDBUrl =
      "mongodb+srv://pratyush-embelia:<your-password-here>@embeliacluster.evgkcdj.mongodb.net/users?retryWrites=true&w=majority";

  static connect() async {
    db = await Db.create(mongoDBUrl);
    await db.open();
    inspect(db);
  }

  static disconnect() async {
    await db.close();
  }

  static doesUserExist(String email) async {
    var collection = await db.collection(email);
    if (collection == null) {
      return false;
    } else {
      return true;
    }
  }

  static insertOne(Map<dynamic, dynamic> data, String email) async {
    var collection = await db.collection(email);
    if (data.isNotEmpty) {
      await collection.insertOne(data);
    }
  }

  static deleteOne(Map<dynamic, dynamic> data, String email) async {
    var collection = await db.collection(email);
    await collection.deleteOne(data);
  }

  static getAll(String email) async {
    var collection = await db.collection(email);
    if (collection == null) {
      return [];
    } else {
      return await collection.find().toList();
    }
  }
}
