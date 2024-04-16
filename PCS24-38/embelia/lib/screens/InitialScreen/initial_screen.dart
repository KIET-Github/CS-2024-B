import 'dart:ui';

import 'package:embelia/buttons/elevated_button/text_input.dart';
import 'package:embelia/localStorage/local_storage.dart';
import 'package:embelia/routes/router.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../authentication/user_auth.dart';

class InitialScreen extends StatefulWidget {
  const InitialScreen({Key? key}) : super(key: key);

  @override
  State<InitialScreen> createState() => _InitialScreenState();
}

class _InitialScreenState extends State<InitialScreen> {
  @override
  void initState() {
    UserAuth().checkLogin().then(
      (value) {
        if (value == true) {
          GoRouter.of(context).goNamed(MyAppRouteConstants.homeScreen);
        } else {
          if (LocalStorage.getString('email') != null) {
            GoRouter.of(context).goNamed(MyAppRouteConstants.homeScreen);
          }
        }
      },
    );
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.of(context).size.height;
    final screenWidth = MediaQuery.of(context).size.width;
    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: Center(
        child: Stack(
          alignment: Alignment.center,
          children: [
            Container(
              height: screenHeight,
              width: screenWidth,
              decoration: const BoxDecoration(
                image: DecorationImage(
                  image: AssetImage('assets/images/soothing_background.jpg'),
                  fit: BoxFit.cover,
                ),
              ),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Container(
                  color: Colors.black.withOpacity(0.5),
                ),
              ),
            ),
            Container(
              height: screenHeight / 1.6,
              width: screenWidth / 1.2,
              decoration: BoxDecoration(
                image: const DecorationImage(
                  filterQuality: FilterQuality.high,
                  opacity: 0.9,
                  image: AssetImage('assets/images/soothing_background.jpg'),
                  fit: BoxFit.cover,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.5),
                    blurRadius: 10,
                    spreadRadius: 5,
                  ),
                ],
                borderRadius: BorderRadius.circular(20),
                color: Colors.white.withOpacity(0.6),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const InputTextButton(), // Email-Password Sign In
                  const SizedBox(
                    height: 10,
                  ),
                  const Divider(
                    endIndent: 20,
                    indent: 20,
                    color: Colors.black,
                    thickness: 2,
                  ),
                  const SizedBox(
                    height: 10,
                  ),
                  Container(
                    height: 50,
                    width: 50,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(30),
                      color: Colors.white.withOpacity(0.8),
                    ),
                    child: IconButton(
                      icon: Image.asset(
                        'assets/images/google_icon.png',
                        height: 40,
                        width: 50,
                      ),
                      onPressed: () async {
                        try {
                          UserAuth.signInWithGoogle().then(
                            (value) async {
                              await LocalStorage.setString(
                                      'email', UserAuth().userEmail)
                                  .then(
                                (value) => LocalStorage.setString(
                                        'name', UserAuth().userName)
                                    .then(
                                  (value) {
                                    Navigator.pop(context);
                                    GoRouter.of(context).goNamed(
                                        MyAppRouteConstants.homeScreen);
                                  },
                                ),
                              );
                            },
                          );
                        } catch (e) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text(
                                e.toString(),
                                style: const TextStyle(
                                  fontSize: 20,
                                ),
                              ),
                            ),
                          );
                        }
                      },
                    ),
                  ),
                  Text(
                    'Sign In with Google',
                    style: GoogleFonts.ubuntu(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
