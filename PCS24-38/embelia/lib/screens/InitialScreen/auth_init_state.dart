part of 'auth_init_cubit.dart';

@immutable
abstract class AuthInitState {}

class AuthInitInitial extends AuthInitState {}

class AuthInitSignInLoading extends AuthInitState{}

class AuthInitSignedIn extends AuthInitState{}

class AuthInitSignedInError extends AuthInitState{
  final String error;
  AuthInitSignedInError(this.error);
}

class AuthInitSignUpLoading extends AuthInitState{}

class AuthInitSignedUp extends AuthInitState{}

class AuthInitSignedUpError extends AuthInitState{
  final String error;
  AuthInitSignedUpError(this.error);
}
