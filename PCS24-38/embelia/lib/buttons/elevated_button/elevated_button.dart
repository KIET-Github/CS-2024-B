import 'package:flutter/material.dart';

class CustomElevatedButton extends StatelessWidget {
  const CustomElevatedButton(
      {super.key,
      required this.myFunc,
      required this.myText,
      required this.icon});
  final Function myFunc;
  final String myText;
  final Widget icon;

  @override
  Widget build(BuildContext context) {
    final ScreenWidth = MediaQuery.of(context).size.width;
    return SizedBox(
      width: ScreenWidth / 2,
      height: 50,
      child: ElevatedButton.icon(
        icon: icon,
        onPressed: () {
          myFunc();
        },
        style: ElevatedButton.styleFrom(
          elevation: 0,
          backgroundColor: Colors.deepOrange[500],
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
        label: Text(
          myText,
          style: const TextStyle(color: Colors.black),
        ),
      ),
    );
  }
}

class RoundedElevatedButton extends StatelessWidget {
  const RoundedElevatedButton(
      {super.key, required this.child, required this.onPressed});
  final Text child;
  final Function onPressed;

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.all(
          Radius.circular(20),
        )),
        foregroundColor: Colors.black,
        backgroundColor: Colors.white,
        shadowColor: Colors.red,
        elevation: 5,
      ),
      onPressed: () => onPressed(),
      child: child,
    );
  }
}
