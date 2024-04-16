import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../utils/constants.dart';
import '../user_auth.dart';
import '../user_data_schema.dart';

part 'user_data_firebase_state.dart';

class UserDataFirebaseCubit extends Cubit<UserDataFirebaseState> {
  UserDataFirebaseCubit() : super(UserDataFirebaseInitial());

  final data = FirebaseFirestore.instance.collection('users');

  int healthScore = 0;
  int totalTask = 0;
  int streak = 0;
  int prizeMoney = 0;
  String taskName = '';
  bool isTaskComplete = false;
  dynamic time;
  dynamic currentTime;

  setTaskName(String value) {
    taskName = value;
    emit(UserDataFirebaseInitial());
  }

  setTaskComplete(bool value) {
    isTaskComplete = value;
    emit(UserDataFirebaseInitial());
  }

  setTime(dynamic value) {
    time = value;
    emit(UserDataFirebaseInitial());
  }

  setCurrentTime(dynamic value) {
    currentTime = value;
    emit(UserDataFirebaseInitial());
  }

  Future<bool> doesUserExist(BuildContext context) async {
    DocumentSnapshot snapshot = await data.doc(UserAuth().userEmail).get();
    return snapshot.exists;
  }

  Future<bool> doesUserHealthProfileExist(BuildContext context) async {
    DocumentSnapshot snapshot = await data.doc(UserAuth().userEmail).get();
    return snapshot['healthProfileCreated'];
  }

  Future init(BuildContext context) async {
    emit(UserDataFirebaseLoading());

    await doesUserExist(context).then(
      (value) {
        if (value == true) {
          getUserData(context).then(
            (value) {
              emit(
                UserDataFirebaseLoaded(
                  healthScore: healthScore,
                  totalTask: totalTask,
                  streak: streak,
                  prizeMoney: prizeMoney,
                  taskName: taskName,
                  isTaskComplete: isTaskComplete,
                  time: time,
                  currentTime: currentTime,
                ),
              );
            },
          );
        } else {
          createUserData(context).then(
            (value) {
              emit(
                UserDataFirebaseLoaded(
                  healthScore: healthScore,
                  totalTask: totalTask,
                  streak: streak,
                  prizeMoney: prizeMoney,
                  taskName: taskName,
                  isTaskComplete: isTaskComplete,
                  time: time,
                  currentTime: currentTime,
                ),
              );
            },
          );
        }
      },
    );
  }

  Future createUserData(BuildContext context) async {
    final userData = UserDataSchema(
      email: UserAuth().userEmail,
      healthScore: 0,
      streak: 0,
      prizeMoney: 0,
      name: UserAuth().userName,
      totalTask: 0,
      healthProfileCreated: false,
    );

    final json = userData.toMap();
    await data.doc(UserAuth().userEmail).set(json);
  }

  Future getUserData(BuildContext context) async {
    DocumentSnapshot snapshot = await data.doc(UserAuth().userEmail).get();
    healthScore = snapshot['healthScore'];
    streak = snapshot['streak'];
    prizeMoney = snapshot['prizeMoney'];
    totalTask = snapshot['totalTask'];
  }

  Future addUserTaskData(
      BuildContext context, Map<String, dynamic> taskData) async {
    emit(UserDataFirebaseLoading());
    if (taskName.isNotEmpty) {
      await data
          .doc(taskData['email'])
          .collection('totalTask')
          .doc(taskName)
          .set(taskData);
      totalTask += 1;
      await data.doc(taskData['email']).update({
        'totalTask': totalTask,
      });
      emit(
        UserDataFirebaseLoaded(
          healthScore: healthScore,
          totalTask: totalTask,
          streak: streak,
          prizeMoney: prizeMoney,
          taskName: taskName,
          isTaskComplete: isTaskComplete,
          time: time,
          currentTime: currentTime,
        ),
      );
    } else {
      logger.e('Task Name is Empty');
    }
  }

  Future deleteUserTaskData(BuildContext context, String taskName) async {
    emit(UserDataFirebaseLoading());
    totalTask -= 1;
    currentTime = DateTime.now().millisecondsSinceEpoch;
    // get taskStartTime of the task to be deleted
    DocumentSnapshot snapshot = await data
        .doc(UserAuth().userEmail)
        .collection('totalTask')
        .doc(taskName)
        .get();
    time = snapshot['taskStartTime'];
    final totalExpectedTime = snapshot['taskTotalTime'];
    final value = (currentTime - time) / 3600000;
    if (value < totalExpectedTime) {
      healthScore += 2;
      prizeMoney += 10;
      streak = 0;
    } else {
      healthScore -= 2;
      streak -= 1;
      prizeMoney -= 50;
    }

    // Delete Task Data from Firebase
    await data
        .doc(UserAuth().userEmail)
        .collection('totalTask')
        .doc(taskName)
        .delete();

    // Update User Data in Firebase
    await data.doc(UserAuth().userEmail).update({
      'totalTask': totalTask,
      'healthScore': healthScore,
      'streak': streak,
      'prizeMoney': prizeMoney,
    });

    // Update User Data in Cubit
    emit(
      UserDataFirebaseLoaded(
        healthScore: healthScore,
        totalTask: totalTask,
        streak: streak,
        prizeMoney: prizeMoney,
        taskName: taskName,
        isTaskComplete: isTaskComplete,
        time: time,
        currentTime: currentTime,
      ),
    );
  }
}
