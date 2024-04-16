part of 'auth_cubit.dart';

@immutable
abstract class AuthState {}

class AuthInitial extends AuthState {}

class AuthLoading extends AuthState {}

class AuthSignOut extends AuthState {}

class AuthSignIn extends AuthState {}

class AuthError extends AuthState {
  final String message;
  AuthError(this.message);
}
