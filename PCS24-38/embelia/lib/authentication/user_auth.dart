import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';

import '../localStorage/local_storage.dart';

class UserAuth extends ChangeNotifier {
  static GoogleSignInAccount? _user;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  static String id = "UserAuth";

  dynamic get userEmail => _user?.email ?? LocalStorage.getString('email');

  Future<void> getUserName() async {
    await FirebaseFirestore.instance
        .collection('users')
        .doc(_user?.email ?? LocalStorage.getString('email'))
        .get()
        .then((value) async {
      await LocalStorage.setString('name', value['name']);
    });
  }

  dynamic get userName => _user?.email ?? LocalStorage.getString('name');

  set user(GoogleSignInAccount? user) {
    _user = user;
  }

  static GoogleSignIn googleSignIn = GoogleSignIn(
    scopes: [
      'email',
      'https://www.googleapis.com/auth/contacts.readonly',
    ],
  );

  static signInWithGoogle() async {
    try {
      _user = await googleSignIn.signIn();
    } catch (e) {
      return e;
    }
  }

  Future<bool> checkLogin() async => await googleSignIn.isSignedIn();

  Future signOutFromGoogle() async {
    if (await checkLogin()) {
      await googleSignIn.signOut();
    }
  }
}
