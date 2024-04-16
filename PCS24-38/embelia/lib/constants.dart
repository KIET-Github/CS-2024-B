import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';

const uri =
    "http://ec2-13-48-44-27.eu-north-1.compute.amazonaws.com:8080/api/FAQ";

extension StringCasingExtension on String {
  String toCapitalized() =>
      length > 0 ? '${this[0].toUpperCase()}${substring(1).toLowerCase()}' : '';
  String toTitleCase() => replaceAll(RegExp(' +'), ' ')
      .split(' ')
      .map((str) => str.toCapitalized())
      .join(' ');
}

const kGradient = LinearGradient(
  colors: <Color>[
    Color(0xff000000),
    Color(0xff150050),
    Color(0xff000000),
    Color(0xff000000),
  ],
);

class MyColor {
  static const Color primaryColor = Color(0xfff9e7ed);
  static const Color secondaryColor = Color(0xff150050);
  static const Color tertiaryColor = Color(0xff000000);
  static const Color quaternaryColor = Color(0xff000000);
  static const Color lightGreyShade = Color(0xffE6EDf5);
}

class MyColorHexCode {
  static const int emeraldGreen = 0xff50C878;
  static const int cerise = 0xffDE3163;
}

GoogleSignIn googleSignIn = GoogleSignIn();
