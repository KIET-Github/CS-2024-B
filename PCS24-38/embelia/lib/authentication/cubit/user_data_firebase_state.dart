part of 'user_data_firebase_cubit.dart';

@immutable
abstract class UserDataFirebaseState {}

class UserDataFirebaseInitial extends UserDataFirebaseState {}

class UserDataFirebaseLoading extends UserDataFirebaseState {}

class UserDataFirebaseLoaded extends UserDataFirebaseState {
  final int healthScore;
  final int totalTask;
  final int streak;
  final int prizeMoney;
  final String taskName;
  final bool isTaskComplete;
  final dynamic time;
  final dynamic currentTime;

  UserDataFirebaseLoaded({
    required this.healthScore,
    required this.totalTask,
    required this.streak,
    required this.prizeMoney,
    required this.taskName,
    required this.isTaskComplete,
    required this.time,
    required this.currentTime,
  });
}

class UserDataFirebaseError extends UserDataFirebaseState {
  final String message;

  UserDataFirebaseError({required this.message});
}

class UserDataFirebaseUpdated extends UserDataFirebaseState {
  final int healthScore;
  final int totalTask;
  final int streak;
  final int prizeMoney;
  final String taskName;
  final bool isTaskComplete;
  final dynamic time;
  final dynamic currentTime;

  UserDataFirebaseUpdated({
    required this.healthScore,
    required this.totalTask,
    required this.streak,
    required this.prizeMoney,
    required this.taskName,
    required this.isTaskComplete,
    required this.time,
    required this.currentTime,
  });
}
