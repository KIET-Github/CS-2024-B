// ignore_for_file: public_member_api_docs, sort_constructors_first
import 'dart:convert';

class DataSchema {
  final String name;
  final String email;
  final String taskName;
  final int taskStartTime;
  final double taskTotalTime;

  DataSchema(
      {required this.name,
      required this.email,
      required this.taskName,
      required this.taskStartTime,
      required this.taskTotalTime});

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'name': name,
      'email': email,
      'taskName': taskName,
      'taskStartTime': taskStartTime,
      'taskTotalTime': taskTotalTime,
    };
  }

  factory DataSchema.fromMap(Map<String, dynamic> map) {
    return DataSchema(
      name: map['name'] as String,
      email: map['email'] as String,
      taskName: map['taskName'] as String,
      taskStartTime: map['taskStartTime'] as int,
      taskTotalTime: map['taskTotalTime'] as double,
    );
  }

  String toJson() => json.encode(toMap());

  factory DataSchema.fromJson(String source) =>
      DataSchema.fromMap(json.decode(source) as Map<String, dynamic>);
}
