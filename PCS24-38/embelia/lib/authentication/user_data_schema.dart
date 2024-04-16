// ignore_for_file: public_member_api_docs, sort_constructors_first
import 'dart:convert';

class UserDataSchema {
  final String email;
  final int healthScore;
  final int streak;
  final int prizeMoney;
  final String name;
  final int totalTask;
  final bool healthProfileCreated;
  UserDataSchema({
    required this.email,
    required this.healthScore,
    required this.streak,
    required this.prizeMoney,
    required this.name,
    required this.totalTask,
    required this.healthProfileCreated,
  });

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'email': email,
      'healthScore': healthScore,
      'streak': streak,
      'prizeMoney': prizeMoney,
      'name': name,
      'totalTask': totalTask,
      'healthProfileCreated': healthProfileCreated,
    };
  }

  factory UserDataSchema.fromMap(Map<String, dynamic> map) {
    return UserDataSchema(
      email: map['email'] as String,
      healthScore: map['healthScore'] as int,
      streak: map['streak'] as int,
      prizeMoney: map['prizeMoney'] as int,
      name: map['name'] as String,
      totalTask: map['totalTask'] as int,
      healthProfileCreated: map['healthProfileCreated'] as bool,
    );
  }

  String toJson() => json.encode(toMap());

  factory UserDataSchema.fromJson(String source) =>
      UserDataSchema.fromMap(json.decode(source) as Map<String, dynamic>);
}
