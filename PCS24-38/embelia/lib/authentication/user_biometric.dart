import 'package:flutter/services.dart';
import 'package:local_auth/local_auth.dart';

class LocalAuth {
  static final auth = LocalAuthentication();
  static Future<bool> hasBiometrics() async {
    try {
      return await auth.canCheckBiometrics;
    } on PlatformException {
      return false;
    }
  }

  static Future<bool> authenticate() async {
    final isAvailable = await hasBiometrics();
    if (!isAvailable) {
      return false;
    }

    try {
      return await auth.authenticate(
        localizedReason: 'Scan to continue',
        options: const AuthenticationOptions(
            useErrorDialogs: true, stickyAuth: true),
      );
    } on PlatformException {
      return false;
    }
  }
}
