import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../screens/InitialScreen/auth_init_cubit.dart';

class InputTextButton extends StatefulWidget {
  static const id = "InputTextButton";

  const InputTextButton({super.key});

  @override
  State<InputTextButton> createState() => _InputTextButtonState();
}

class _InputTextButtonState extends State<InputTextButton> {
  List<bool> isSelected = [true, false];
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _nameController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        ToggleButtons(
          borderRadius: BorderRadius.circular(10),
          borderColor: Colors.black,
          fillColor: Colors.orange[100],
          onPressed: (int index) {
            setState(
              () {
                for (int buttonIndex = 0;
                    buttonIndex < isSelected.length;
                    buttonIndex++) {
                  if (buttonIndex == index) {
                    isSelected[buttonIndex] = true;
                  } else {
                    isSelected[buttonIndex] = false;
                  }
                }
              },
            );
          },
          isSelected: isSelected,
          children: [
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                'Sign In',
                style: GoogleFonts.ubuntu(
                  fontSize: 20,
                  color: Colors.black,
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                'Sign Up',
                style: GoogleFonts.ubuntu(
                  fontSize: 20,
                  color: Colors.black,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(
          height: 20,
        ),
        if (isSelected[0])
          Form(
            key: _formKey,
            child: Column(
              children: [
                InputButton(
                  onValidate: (value) {
                    return (value!.isEmpty ||
                            !RegExp(r'^[\w-.]+@([\w-]+\.)+[\w-]{2,4}$')
                                .hasMatch(value))
                        ? 'Enter correct email'
                        : null;
                  },
                  controller: _emailController,
                  hintText: '',
                  keyboardType: TextInputType.emailAddress,
                  icon: Icons.email,
                  labelText: 'Email *',
                  obsecureText: false,
                ),
                InputButton(
                  onValidate: (value) {
                    if (value!.isEmpty) {
                      return 'Enter correct Password';
                    } else if (value.length < 8) {
                      return "Password should be at least 8 characters";
                    } else {
                      return null;
                    }
                  },
                  controller: _passwordController,
                  hintText: '',
                  keyboardType: TextInputType.name,
                  icon: Icons.password,
                  labelText: 'Password *',
                  obsecureText: true,
                ),
                const SizedBox(
                  height: 10,
                ),
                ElevatedButton(
                  onPressed: () async {
                    if (_formKey.currentState!.validate()) {
                      BlocProvider.of<AuthInitCubit>(context)
                          .signInUsingEmailPassword(
                        email: _emailController.text,
                        password: _passwordController.text,
                        context: context,
                      );
                    }
                  },
                  style: ElevatedButton.styleFrom(
                    minimumSize: const Size(200, 50),
                    backgroundColor: Colors.orange[100],
                    elevation: 0,
                    side: const BorderSide(color: Colors.black, width: 1.5),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  child: BlocBuilder<AuthInitCubit, AuthInitState>(
                    builder: (context, state) {
                      if (state is AuthInitSignInLoading) {
                        return Text(
                          'Loading...',
                          style: GoogleFonts.roboto(
                              fontSize: 20, color: Colors.green),
                        );
                      } else if (state is AuthInitSignedInError) {
                        return Text(
                          state.error,
                          style: GoogleFonts.roboto(
                              fontSize: 20, color: Colors.red),
                        );
                      } else {
                        return Text(
                          'Submit',
                          style: GoogleFonts.roboto(
                              fontSize: 20, color: Colors.black),
                        );
                      }
                    },
                  ),
                ),
              ],
            ),
          )
        else
          Form(
            key: _formKey,
            child: Column(
              children: [
                InputButton(
                  onValidate: (value) {
                    return (value!.isEmpty ||
                            !RegExp(r'^[a-z A-Z]+$').hasMatch(value))
                        ? 'Enter correct name'
                        : null;
                  },
                  controller: _nameController,
                  hintText: 'What do people call you?',
                  keyboardType: TextInputType.name,
                  icon: Icons.person,
                  labelText: 'Name *',
                  obsecureText: false,
                ),
                InputButton(
                  onValidate: (value) {
                    return (value!.isEmpty ||
                            !RegExp(r'^[\w-.]+@([\w-]+\.)+[\w-]{2,4}$')
                                .hasMatch(value))
                        ? 'Enter correct email'
                        : null;
                  },
                  controller: _emailController,
                  hintText: 'Where do people email you?',
                  keyboardType: TextInputType.emailAddress,
                  icon: Icons.email,
                  labelText: 'Email *',
                  obsecureText: false,
                ),
                InputButton(
                  onValidate: (value) {
                    if (value!.isEmpty) {
                      return 'Enter correct name';
                    } else if (value.length < 8) {
                      return "Password should be at least 8 characters";
                    } else {
                      return null;
                    }
                  },
                  controller: _passwordController,
                  hintText: 'Enter your password',
                  keyboardType: TextInputType.name,
                  icon: Icons.password,
                  labelText: 'Password *',
                  obsecureText: true,
                ),
                const SizedBox(
                  height: 10,
                ),
                ElevatedButton(
                  onPressed: () async {
                    if (_formKey.currentState!.validate()) {
                      BlocProvider.of<AuthInitCubit>(context)
                          .signUpUsingEmailPassword(
                        email: _emailController.text,
                        password: _passwordController.text,
                        context: context,
                        name: _nameController.text,
                      );
                    }
                  },
                  style: ElevatedButton.styleFrom(
                    minimumSize: const Size(200, 50),
                    backgroundColor: Colors.orange[100],
                    elevation: 0,
                    side: const BorderSide(color: Colors.black, width: 1.5),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  child: BlocBuilder<AuthInitCubit, AuthInitState>(
                    builder: (context, state) {
                      if (state is AuthInitSignUpLoading) {
                        return Text(
                          'Loading...',
                          style: GoogleFonts.roboto(
                              fontSize: 20, color: Colors.green),
                        );
                      } else if (state is AuthInitSignedInError) {
                        return Text(
                          state.error,
                          style: GoogleFonts.roboto(
                              fontSize: 20, color: Colors.red),
                        );
                      }
                      return Text(
                        'Submit',
                        style: GoogleFonts.roboto(
                            fontSize: 20, color: Colors.black),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }
}

class InputButton extends StatefulWidget {
  const InputButton(
      {super.key,
      required this.onValidate,
      required this.hintText,
      required this.obsecureText,
      required this.keyboardType,
      required this.icon,
      required this.labelText,
      required this.controller});

  final String? Function(String?) onValidate;
  final String hintText;
  final String labelText;
  final TextInputType keyboardType;
  final IconData icon;
  final bool obsecureText;
  final TextEditingController controller;

  @override
  State<InputButton> createState() => _InputButtonState();
}

class _InputButtonState extends State<InputButton> {
  bool obsecureText = false;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 15.0, vertical: 5),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.grey[200],
          border: Border.all(color: Colors.black, width: 1),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Padding(
          padding: const EdgeInsets.only(left: 20),
          child: TextFormField(
            controller: widget.controller,
            obscureText: obsecureText,
            cursorColor: Colors.deepOrange,
            keyboardType: widget.keyboardType,
            decoration: InputDecoration(
              border: InputBorder.none,
              suffixIcon: widget.obsecureText
                  ? IconButton(
                      onPressed: () {
                        setState(() {
                          obsecureText = !obsecureText;
                        });
                      },
                      icon: Icon(
                        obsecureText ? Icons.visibility : Icons.visibility_off,
                        color: Colors.black,
                      ),
                    )
                  : null,
              hintText: widget.hintText,
              labelText: widget.labelText,
              labelStyle: const TextStyle(color: Colors.black),
              icon: Icon(
                widget.icon,
                color: Colors.black,
                size: 20,
              ),
            ),
            // onSaved: onSave,
            validator: widget.onValidate,
          ),
        ),
      ),
    );
  }
}
