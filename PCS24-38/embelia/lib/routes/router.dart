import 'package:embelia/screens/InitialScreen/initial_screen.dart';
import 'package:embelia/screens/chat_bot.dart';
import 'package:embelia/screens/homeScreen/home_screen.dart';
import 'package:embelia/screens/user_health_profile.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../screens/faq/faq.dart';

final GoRouter router = GoRouter(
  routes: <RouteBase>[
    GoRoute(
      name: '/',
      path: MyAppRouteConstants.initialScreen,
      pageBuilder: (BuildContext context, GoRouterState state) {
        return CustomTransitionPage(
          child: const InitialScreen(),  // <--- Change this to InitialScreen()
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            var begin = const Offset(0.0, 1.0);
            var end = Offset.zero;
            var curve = Curves.ease;
            var tween =
                Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
            return SlideTransition(
              position: animation.drive(tween),
              child: child,
            );
          },
        );
      },
    ),
    GoRoute(
      name: '/homeScreen',
      path: MyAppRouteConstants.homeScreen,
      pageBuilder: (BuildContext context, GoRouterState state) {
        return CustomTransitionPage(
          child: const HomeScreen(),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            var begin = const Offset(0.0, 1.0);
            var end = Offset.zero;
            var curve = Curves.ease;
            var tween =
                Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
            return SlideTransition(
              position: animation.drive(tween),
              child: child,
            );
          },
        );
      },
    ),
    GoRoute(
      name: '/userHeathProfile',
      path: MyAppRouteConstants.userHeathProfile,
      pageBuilder: (BuildContext context, GoRouterState state) {
        return CustomTransitionPage(
          child: const UserHealthProfile(),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            var begin = const Offset(0.0, 1.0);
            var end = Offset.zero;
            var curve = Curves.ease;
            var tween =
                Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
            return SlideTransition(
              position: animation.drive(tween),
              child: child,
            );
          },
        );
      },
    ),
    GoRoute(
      name: '/kFAQ',
      path: MyAppRouteConstants.kFAQ,
      pageBuilder: (BuildContext context, GoRouterState state) {
        return CustomTransitionPage(
          child: const FAQ(),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            var begin = const Offset(0.0, 1.0);
            var end = Offset.zero;
            var curve = Curves.ease;
            var tween =
                Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
            return SlideTransition(
              position: animation.drive(tween),
              child: child,
            );
          },
        );
      },
    ),
    GoRoute(
      name: '/chatBotScreen',
      path: MyAppRouteConstants.chatBotScreen,
      pageBuilder: (BuildContext context, GoRouterState state) {
        return CustomTransitionPage(
          child: const ChatBot(),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            var begin = const Offset(0.0, 1.0);
            var end = Offset.zero;
            var curve = Curves.ease;
            var tween =
                Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
            return SlideTransition(
              position: animation.drive(tween),
              child: child,
            );
          },
        );
      },
    ),
  ],
);

// Define constants for route names
class MyAppRouteConstants {
  static const String initialScreen = '/';
  static const String homeScreen = '/homeScreen';
  static const String userHeathProfile = '/userHeathProfile';
  static const String kFAQ = '/kFAQ';
  static const String chatBotScreen = '/chatBotScreen';
}
