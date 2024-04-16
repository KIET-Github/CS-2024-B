import 'package:bloc/bloc.dart';
import 'package:embelia/localStorage/local_storage.dart';
import 'package:embelia/routes/router.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

part 'auth_init_state.dart';

class AuthInitCubit extends Cubit<AuthInitState> {
  AuthInitCubit() : super(AuthInitInitial());

  final FirebaseAuth _auth = FirebaseAuth.instance;

  Future<void> signUpUsingEmailPassword(
      {required String email,
      required String password,
      required BuildContext context,
      required String name}) async {
    emit(AuthInitSignUpLoading());
    await _auth
        .createUserWithEmailAndPassword(email: email, password: password)
        .then((value) async {
      emit(AuthInitSignedUp());
      await LocalStorage.setString('email', email).then((value) async =>
          await LocalStorage.setString('name', name).then((value) async =>
              GoRouter.of(context).goNamed(MyAppRouteConstants.homeScreen)));
    }).catchError((e) {
      switch (e.code) {
        case 'email-already-in-use':
          emit(AuthInitSignedUpError('Email already in use'));
          Future.delayed(const Duration(seconds: 2), () {
            emit(AuthInitInitial());
          });
          break;
        case 'invalid-email':
          emit(AuthInitSignedUpError('Invalid email'));
          Future.delayed(const Duration(seconds: 2), () {
            emit(AuthInitInitial());
          });
          break;
        case 'weak-password':
          emit(AuthInitSignedUpError('Weak password'));
          Future.delayed(const Duration(seconds: 2), () {
            emit(AuthInitInitial());
          });
          break;
        default:
          emit(AuthInitSignedUpError('Something went wrong'));
          Future.delayed(const Duration(seconds: 2), () {
            emit(AuthInitInitial());
          });
      }
    });
  }

  Future<void> signInUsingEmailPassword(
      {required String email,
      required String password,
      required BuildContext context}) async {
    emit(AuthInitSignInLoading());
    await _auth
        .signInWithEmailAndPassword(email: email, password: password)
        .then((value) async {
      emit(AuthInitSignedIn());
      await LocalStorage.setString('email', email).then((value) async =>
          GoRouter.of(context).goNamed(MyAppRouteConstants.homeScreen));
    }).catchError((e) {
      switch (e.code) {
        case 'user-not-found':
          emit(AuthInitSignedInError('User not found'));
          Future.delayed(const Duration(seconds: 2), () {
            emit(AuthInitInitial());
          });
          break;
        case 'wrong-password':
          emit(AuthInitSignedInError('Wrong password'));
          Future.delayed(const Duration(seconds: 2), () {
            emit(AuthInitInitial());
          });
          break;
        default:
          emit(AuthInitSignedInError('Something went wrong'));
          Future.delayed(const Duration(seconds: 2), () {
            emit(AuthInitInitial());
          });
      }
    });
  }
}
