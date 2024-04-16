import 'package:bloc/bloc.dart';
import 'package:flutter/cupertino.dart';
import 'package:go_router/go_router.dart';

import '../../authentication/user_auth.dart';
import '../../localStorage/local_storage.dart';
import '../../routes/router.dart';

part 'auth_state.dart';

class AuthCubit extends Cubit<AuthState> {
  AuthCubit() : super(AuthInitial());

  Future signOut(BuildContext context) async {
    emit(AuthLoading());
    UserAuth().signOutFromGoogle().then(
      (value) async {
        await LocalStorage.clear().then(
          (value) {
            Future.delayed(const Duration(seconds: 2)).then((value) {
              GoRouter.of(context).goNamed(MyAppRouteConstants.initialScreen);
              emit(AuthSignOut());
            }, onError: (e) {
              emit(AuthError(e.toString()));
            });
          },
        );
      },
    );
  }
}
